import os
from typing import List, Dict, Type
import math
from datetime import datetime

import torch
from torch.optim import Optimizer
import transformers

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from torch.nn import CrossEntropyLoss
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np

from medblip.utils import reduce_and_visualize_multi, TrainingVisualizer, eval_metrics

WEIGHTS_NAME = "pytorch_model.bin"

import shutil

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None, phase_name=None, training_mode=None):
        self.phase_name = phase_name
        self.training_mode = training_mode  # 1=教师模型, 2=学生模型, 3=两步都执行
        self.visualizer = TrainingVisualizer(phase_name=phase_name) #realtime plot

    def train(self,
        model,
        dataloader,
        valdataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = 'Alifuse_bibm/checkpoints',
        metric_path: str = '',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        accumulation_steps: int = 1,
        patience: int = 5,
        save_interval: int = 0,  # 控制检查点保存频率，0表示不保存每个epoch的检查点
        # Model selection / early stopping knobs
        min_delta_auc: float = 0.0,          # AUC提升超过该阈值才认为“更好”
        min_delta_loss: float = 0.0,         # loss下降超过该阈值才认为“更好”
        tie_breaker_eps: float = 1e-4,       # AUC差距小于该值时，用更低val loss作为平局裁决
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        
        # 删除整个checkpoint目录（连目录一起删再重建）
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)

        self.accumulation_steps = accumulation_steps
        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # 检查是否需要冻结教师模型参数
        freeze_teacher = hasattr(model, 'freeze_teacher') and model.freeze_teacher

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        # 根据是否冻结教师模型参数调整优化器参数组
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()
        skip_scheduler = False
        
        # 打印训练模式信息
        if hasattr(model, 'freeze_teacher') and model.freeze_teacher:
            print(f"\n===== 训练模式: 冻结教师模型，只训练学生模型 =====")
        else:
            print(f"\n===== 训练模式: 同时训练教师模型和学生模型 =====")
            
        # 早停机制参数
        best_f1 = -1.0
        best_val_loss_for_best_f1 = float("inf")  # 仅用于F1平局裁决（跟随best_f1更新）
        best_val_loss_for_early_stop = float("inf")  # 仅用于早停（独立跟踪最小val loss）
        no_improve_epochs = 0  # 基于val loss的早停计数
        early_stop_triggered = False
        best_epoch = -1
        
        # 在训练开始前进行一次预评估，了解初始模型性能
        print("\n===== 训练前预评估 =====")
        initial_teacher_auc, initial_teacher_acc, initial_student_auc, initial_student_acc, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = self.test(model, valdataloader, metric_path, -1)
        print("=====================")
        
        for epoch in range(epochs):
            if early_stop_triggered:
                print(f"[{self.phase_name if self.phase_name else '训练'}] 早停机制触发，停止训练")
                break
            # 1. 先进行训练
            data_iterator = iter(dataloader)
            epoch_train_loss, epoch_loss_itc, epoch_loss_text_res, epoch_loss_image_res, epoch_loss_cog_res, epoch_loss_cls = 0,0,0,0,0,0
            epoch_loss_cls_tea, epoch_loss_cls_stu, epoch_loss_kl, epoch_loss_feat = 0.,0.,0.,0.  # 蒸馏相关损失

            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)

                loss = model(data)
                loss_value = loss['loss'] / self.accumulation_steps
                #loss分项
                loss_itc = loss.get('loss_itc', torch.tensor(0.0))
                loss_text_res = loss.get('loss_text_res', torch.tensor(0.0))
                loss_image_res = loss.get('loss_image_res', torch.tensor(0.0))
                loss_cog_res = loss.get('loss_cog_res', torch.tensor(0.0))
                loss_cls = loss.get('loss_cls', torch.tensor(0.0))
                loss_cls_tea = loss.get('loss_cls_teacher', torch.tensor(0.0))
                loss_cls_stu = loss.get('loss_cls_student', torch.tensor(0.0))
                loss_kl = loss.get('loss_kl', torch.tensor(0.0))
                loss_feat = loss.get('loss_feat', torch.tensor(0.0))

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                # 安全处理损失值，确保兼容浮点数和张量
                epoch_train_loss += loss_value.item() if hasattr(loss_value, 'item') else loss_value
                epoch_loss_itc += loss_itc.item() if hasattr(loss_itc, 'item') else loss_itc
                epoch_loss_text_res += loss_text_res.item() if hasattr(loss_text_res, 'item') else loss_text_res
                epoch_loss_image_res += loss_image_res.item() if hasattr(loss_image_res, 'item') else loss_image_res
                epoch_loss_cog_res += loss_cog_res.item() if hasattr(loss_cog_res, 'item') else loss_cog_res
                epoch_loss_cls += loss_cls.item() if hasattr(loss_cls, 'item') else loss_cls
                epoch_loss_cls_tea += loss_cls_tea.item() if hasattr(loss_cls_tea, 'item') else loss_cls_tea
                epoch_loss_cls_stu += loss_cls_stu.item() if hasattr(loss_cls_stu, 'item') else loss_cls_stu
                epoch_loss_kl += loss_kl.item() if hasattr(loss_kl, 'item') else loss_kl
                epoch_loss_feat += loss_feat.item() if hasattr(loss_feat, 'item') else loss_feat

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

            # 计算平均训练损失并更新图表
            avg_train_loss = epoch_train_loss / steps_per_epoch
            avg_loss_itc = epoch_loss_itc / steps_per_epoch
            avg_loss_text_res = epoch_loss_text_res / steps_per_epoch
            avg_loss_image_res = epoch_loss_image_res / steps_per_epoch
            avg_loss_cog_res = epoch_loss_cog_res / steps_per_epoch
            avg_loss_cls = epoch_loss_cls / steps_per_epoch
            avg_loss_cls_tea = epoch_loss_cls_tea / steps_per_epoch
            avg_loss_cls_stu = epoch_loss_cls_stu / steps_per_epoch
            avg_loss_kl = epoch_loss_kl / steps_per_epoch
            avg_loss_feat = epoch_loss_feat / steps_per_epoch

            # 在每个epoch结束时打印训练损失信息
            loss_info = f'Epoch[{epoch}/{epochs}]: loss: {avg_train_loss:.4f}'
            loss_info += f', itc: {avg_loss_itc:.4f}, text_res: {avg_loss_text_res:.4f}'
            loss_info += f', image_res: {avg_loss_image_res:.4f}, cog_res: {avg_loss_cog_res:.4f}, cls: {avg_loss_cls:.4f}'
            
            # 添加蒸馏相关损失信息
            if avg_loss_kl > 0 or avg_loss_cls_tea > 0 or avg_loss_cls_stu > 0 or avg_loss_feat > 0:
                loss_info += f', kl: {avg_loss_kl:.4f}'
                loss_info += f', tea_cls: {avg_loss_cls_tea:.4f}'
                loss_info += f', stu_cls: {avg_loss_cls_stu:.4f}'
                loss_info += f', feat: {avg_loss_feat:.4f}'
            
            print(loss_info)
            
            # 2. 训练完成后再进行评估
            val_teacher_auc, val_teacher_acc, val_student_auc, val_student_acc, val_loss, val_loss_itc, val_loss_text_res, val_loss_image_res, val_loss_cog_res, val_loss_cls, val_loss_cls_tea, val_loss_cls_stu, val_loss_kl, val_loss_feat, val_teacher_f1, val_teacher_precision, val_teacher_recall, val_student_f1, val_student_precision, val_student_recall = self.test(model, valdataloader, metric_path, epoch)
            
            # 打印验证集结果
            if self.training_mode != 2:  # 不是只训练学生模型时
                print(f'Epoch[{epoch}/{epochs}] Val Teacher - AUC: {val_teacher_auc:.4f}, ACC: {val_teacher_acc:.4f}, F1: {val_teacher_f1:.4f}, Precision: {val_teacher_precision:.4f}, Recall: {val_teacher_recall:.4f}, Loss: {val_loss:.4f}')
            if self.training_mode != 1:  # 不是只训练教师模型时
                print(f'Epoch[{epoch}/{epochs}] Val Student - AUC: {val_student_auc:.4f}, ACC: {val_student_acc:.4f}, F1: {val_student_f1:.4f}, Precision: {val_student_precision:.4f}, Recall: {val_student_recall:.4f}')
            
            # 计算蒸馏效果提升
            if self.training_mode != 1 and self.training_mode != 2:  # 两步都执行时
                auc_improvement = val_student_auc - val_teacher_auc
                acc_improvement = val_student_acc - val_teacher_acc
                print(f'Epoch[{epoch}/{epochs}] Student vs Teacher - AUC Δ: {auc_improvement:+.4f}, ACC Δ: {acc_improvement:+.4f}')
            
            # 选择当前用于“保存最佳模型”的F1分数（越大越好）
            current_f1 = val_teacher_f1 if self.training_mode == 1 else val_student_f1

            # 1) 保存最佳模型：主指标F1；F1接近时用更低val loss做平局裁决
            improved_f1 = current_f1 > (best_f1 + min_delta_auc)  # 使用相同的阈值
            f1_almost_same = abs(current_f1 - best_f1) <= tie_breaker_eps
            improved_tie = f1_almost_same and (val_loss < (best_val_loss_for_best_f1 - min_delta_loss))

            if improved_f1 or improved_tie:
                best_f1 = float(current_f1)
                best_val_loss_for_best_f1 = float(val_loss)
                best_epoch = epoch
                best_model_path = os.path.join(output_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"[{self.phase_name if self.phase_name else '训练'}] 最佳模型已保存! "
                    f"轮次: {epoch}, F1: {best_f1:.4f}, ValLoss: {best_val_loss_for_best_f1:.4f}"
                )

            # 2) 早停：监控val loss（越小越好）
            if val_loss < (best_val_loss_for_early_stop - min_delta_loss):
                best_val_loss_for_early_stop = float(val_loss)
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                print(f"[{self.phase_name if self.phase_name else '训练'}] Val loss 未下降 ({no_improve_epochs}/{patience})")

            if no_improve_epochs >= patience:
                early_stop_triggered = True
                print(f"[{self.phase_name if self.phase_name else '训练'}] 早停触发: {patience}轮 val loss 未下降")
                break

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            # 提取特征用于可视化（仅在多模态训练时）
            image_features = None
            text_features = None
            labels = None
            
            # 检查模型是否支持多模态训练
            if hasattr(model, 'modalities') and model.modalities == 'both':
                try:
                    # 从验证集中提取一批数据用于特征可视化
                    val_data_iter = iter(valdataloader)
                    val_batch = next(val_data_iter)
                    
                    # 确保数据格式正确
                    if len(val_batch) >= 3:
                        # 提取图像、文本和标签
                        val_image = val_batch[0].cuda()
                        # 添加通道维度，使其与模型期望的格式匹配: [bs, 1, 128, 128, 128]
                        val_image = val_image.unsqueeze(1)
                        val_text = val_batch[1].cuda()
                        val_labels = val_batch[2]
                        
                        # 使用模型的特征提取方法获取特征
                        with torch.no_grad():
                            # 检查模型是否有_extract_features方法
                            if hasattr(model, '_extract_features'):
                                features = model._extract_features(val_image, val_text, augment=False)
                                if 'image_feat' in features and 'text_feat' in features:
                                    image_features = features['image_feat']
                                    text_features = features['text_feat']
                                    labels = val_labels
                            else:
                                # 尝试使用forward方法获取特征
                                # 注意：这可能需要根据模型的具体实现进行调整
                                print("模型没有_extract_features方法，跳过特征可视化")
                except Exception as e:
                    print(f"提取特征时出错: {e}")
            
            # 更新可视化指标，添加学习率参数和特征可视化
            self.visualizer.update_metrics(
                epoch, 
                train_loss=avg_train_loss, 
                loss_itc=avg_loss_itc,
                loss_text_res=avg_loss_text_res,
                loss_image_res=avg_loss_image_res,
                loss_cog_res=avg_loss_cog_res,
                loss_cls=avg_loss_cls,
                loss_cls_tea=avg_loss_cls_tea,
                loss_cls_stu=avg_loss_cls_stu,
                loss_kl=avg_loss_kl,
                loss_feat=avg_loss_feat,
                val_acc=val_teacher_acc, 
                val_auc=val_teacher_auc,
                val_acc_stu=val_student_acc, 
                val_auc_stu=val_student_auc,
                val_loss=val_loss,
                val_loss_itc=val_loss_itc,
                val_loss_text_res=val_loss_text_res,
                val_loss_image_res=val_loss_image_res,
                val_loss_cog_res=val_loss_cog_res,
                val_loss_cls=val_loss_cls,
                val_loss_cls_tea=val_loss_cls_tea,
                val_loss_cls_stu=val_loss_cls_stu,
                val_loss_kl=val_loss_kl,
                val_loss_feat=val_loss_feat,
                lr=current_lr,  # 添加学习率
                image_features=image_features,  # 添加图像特征
                text_features=text_features,  # 添加文本特征
                labels=labels  # 添加标签
            )

            # 3. 最后保存检查点（根据save_interval控制频率）
            if not early_stop_triggered and save_interval > 0 and (epoch + 1) % save_interval == 0:
                self._save_ckpt(model, epoch, output_path)
                print(f"[{self.phase_name if self.phase_name else '训练'}] 检查点已保存 (每{save_interval}个epoch)")
            

    def test(
        self,
        model,
        eval_dataloader,
        metric_path,
        epoch,
        ):
        '''
        测试模型性能，同时评估教师模型和学生模型
        '''
        try:
            steps_per_epoch = len(eval_dataloader)
            model = model.cuda()
            
            # 使用tqdm显示进度条
            from tqdm import tqdm
            gts = []
            preds = []  # 教师模型预测概率
            preds_stu = []  # 学生模型预测概率
            probs_teacher = []  # 保存教师模型的原始概率值
            probs_student = []  # 保存学生模型的原始概率值
            
            # 初始化验证损失累积变量
            val_loss = 0.0
            val_loss_itc = 0.0
            val_loss_text_res = 0.0
            val_loss_image_res = 0.0
            val_loss_cog_res = 0.0
            val_loss_cls = 0.0
            val_loss_cls_tea = 0.0
            val_loss_cls_stu = 0.0
            val_loss_kl = 0.0
            val_loss_feat = 0.0
            val_batch_count = 0

            # 创建数据迭代器
            data_iterator = iter(eval_dataloader)
            for eval_iter in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch} Evaluation"):
                try:
                    model.eval()
                    # 获取数据
                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        # 如果迭代器用完，重新创建
                        data_iterator = iter(eval_dataloader)
                        data = next(data_iterator)
                    
                    with torch.no_grad():
                        # 计算验证损失
                        loss = model(data)
                        # 安全地获取损失值，处理tensor和float类型
                        def get_loss_value(loss_val):
                            if hasattr(loss_val, 'item'):
                                return loss_val.item()
                            return loss_val
                        
                        val_loss += get_loss_value(loss['loss'])
                        val_loss_itc += get_loss_value(loss.get('loss_itc', 0.0))
                        val_loss_text_res += get_loss_value(loss.get('loss_text_res', 0.0))
                        val_loss_image_res += get_loss_value(loss.get('loss_image_res', 0.0))
                        val_loss_cog_res += get_loss_value(loss.get('loss_cog_res', 0.0))
                        val_loss_cls += get_loss_value(loss.get('loss_cls', 0.0))
                        val_loss_cls_tea += get_loss_value(loss.get('loss_cls_teacher', 0.0))
                        val_loss_cls_stu += get_loss_value(loss.get('loss_cls_student', 0.0))
                        val_loss_kl += get_loss_value(loss.get('loss_kl', 0.0))
                        val_loss_feat += get_loss_value(loss.get('loss_feat', 0.0))
                        val_batch_count += 1
                        
                        # 获取预测结果
                        pred_output = model.predict(data)
                        # 确保pred_output包含足够的返回值
                        if len(pred_output) >= 2:
                            pred, pred_stu = pred_output[0], pred_output[1]
                        else:
                            # 如果模型没有返回学生模型预测，使用教师模型预测
                            pred = pred_output[0]
                            pred_stu = pred_output[0].clone()
                            print("警告：模型未返回学生模型预测结果，使用教师模型预测代替")
                    
                    # 保存原始概率值
                    probs_teacher.append(pred.cpu())
                    probs_student.append(pred_stu.cpu())
                    
                    # 安全获取标签 - 注意：数据集返回的是 (image, features, cog_features, label, id)
                    if len(data) > 3 and isinstance(data[3], torch.Tensor):
                        gts.append(data[3])
                    else:
                        print(f"警告：无法获取有效的标签数据，跳过此批次")
                        continue
                        
                except Exception as e:
                    print(f"评估批次 {eval_iter} 出错: {e}")
                    continue
            
            # 检查是否有收集到有效数据
            if not probs_teacher or not gts:
                print("警告: 未收集到有效评估数据")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 合并结果
            try:
                probs_teacher = torch.cat(probs_teacher, dim=0)
                probs_student = torch.cat(probs_student, dim=0)
                gts = torch.cat(gts, dim=0)
            except Exception as e:
                print(f"合并结果时出错: {e}")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 过滤无效标签
            try:
                # 获取有效标签的索引
                valid_indices = torch.nonzero(gts != -100).squeeze()
                if valid_indices.ndim == 0:  # 如果只有一个有效样本
                    valid_indices = valid_indices.unsqueeze(0)
                
                # 确保有有效样本
                if len(valid_indices) == 0:
                    print("警告: 没有找到有效标签")
                    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    
                # 过滤数据
                gts = gts[valid_indices].cpu()
                probs_teacher = probs_teacher[valid_indices].cpu()
                probs_student = probs_student[valid_indices].cpu()
            except Exception as e:
                print(f"过滤无效标签时出错: {e}")
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 获取预测类别
            preds_teacher = probs_teacher.argmax(-1)
            preds_student = probs_student.argmax(-1)
            
            # 调试信息
            if epoch % 10 == 0 or epoch == len(range(100)) - 1:  # 每10个epoch或最后一个epoch打印详细信息
                print(f'Ground Truth Sample: {gts[:5]}')
                print(f'Teacher Predictions Sample: {preds_teacher[:5]}')
                print(f'Student Predictions Sample: {preds_student[:5]}')
                # 打印概率值样例以确认其范围是[0,1]
                print(f'Teacher Probabilities Sample: {probs_teacher[:5]}')
            
            y_true = gts
            y_pred_teacher = preds_teacher
            y_pred_student = preds_student

            # 根据训练模式选择性评估
            if self.training_mode == 1:  # 只训练教师模型
                # 只评估教师模型
                auc_teacher, acc_teacher, f1_teacher, precision_teacher, recall_teacher = eval_metrics(y_true, y_pred_teacher, epoch, metric_path, if_student=False, raw_probs=probs_teacher)
                auc_student, acc_student, f1_student, precision_student, recall_student = auc_teacher, acc_teacher, f1_teacher, precision_teacher, recall_teacher
            elif self.training_mode == 2:  # 只训练学生模型
                # 只评估学生模型
                auc_student, acc_student, f1_student, precision_student, recall_student = eval_metrics(y_true, y_pred_student, epoch, metric_path, if_student=True, raw_probs=probs_student)
                auc_teacher, acc_teacher, f1_teacher, precision_teacher, recall_teacher = auc_student, acc_student, f1_student, precision_student, recall_student
            else:  # 两步都执行或默认情况
                # 教师模型评估 - 使用原始概率值计算AUC
                auc_teacher, acc_teacher, f1_teacher, precision_teacher, recall_teacher = eval_metrics(y_true, y_pred_teacher, epoch, metric_path, if_student=False, raw_probs=probs_teacher)
                # 学生模型评估 - 使用原始概率值计算AUC
                auc_student, acc_student, f1_student, precision_student, recall_student = eval_metrics(y_true, y_pred_student, epoch, metric_path, if_student=True, raw_probs=probs_student)
                
                # 计算学生模型相对于教师模型的提升
                auc_improvement = auc_student - auc_teacher
                acc_improvement = acc_student - acc_teacher
                print(f'Student vs Teacher - AUC Δ: {auc_improvement:+.4f}, ACC Δ: {acc_improvement:+.4f}')
            
            # 保存预测结果到CSV文件
            try:
                import pandas as pd
                # 根据训练阶段决定保存哪些模型的预测结果
                if self.training_mode == 1:  # 只训练教师模型
                    results_df = pd.DataFrame({
                        'sample_id': range(len(y_true)),
                        'ground_truth': y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true,
                        'pred_class': y_pred_teacher.numpy() if isinstance(y_pred_teacher, torch.Tensor) else y_pred_teacher,
                    })
                    
                    # 添加各分类的概率值
                    if probs_teacher.ndim > 1:
                        for i in range(probs_teacher.shape[1]):
                            results_df[f'prob_class_{i}'] = probs_teacher[:, i].numpy() if isinstance(probs_teacher[:, i], torch.Tensor) else probs_teacher[:, i]
                    else:
                        results_df['prob_class_0'] = 1 - probs_teacher.numpy() if isinstance(probs_teacher, torch.Tensor) else 1 - probs_teacher
                        results_df['prob_class_1'] = probs_teacher.numpy() if isinstance(probs_teacher, torch.Tensor) else probs_teacher
                    
                    # 添加预测正确性标记
                    results_df['correct'] = (results_df['ground_truth'] == results_df['pred_class']).astype(int)
                    
                elif self.training_mode == 2:  # 只训练学生模型
                    results_df = pd.DataFrame({
                        'sample_id': range(len(y_true)),
                        'ground_truth': y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true,
                        'pred_class': y_pred_student.numpy() if isinstance(y_pred_student, torch.Tensor) else y_pred_student,
                    })
                    
                    # 添加各分类的概率值
                    if probs_student.ndim > 1:
                        for i in range(probs_student.shape[1]):
                            results_df[f'prob_class_{i}'] = probs_student[:, i].numpy() if isinstance(probs_student[:, i], torch.Tensor) else probs_student[:, i]
                    else:
                        results_df['prob_class_0'] = 1 - probs_student.numpy() if isinstance(probs_student, torch.Tensor) else 1 - probs_student
                        results_df['prob_class_1'] = probs_student.numpy() if isinstance(probs_student, torch.Tensor) else probs_student
                    
                    # 添加预测正确性标记
                    results_df['correct'] = (results_df['ground_truth'] == results_df['pred_class']).astype(int)
                    
                else:  # 两步都执行，同时保存两个模型的预测结果
                    results_df = pd.DataFrame({
                        'sample_id': range(len(y_true)),
                        'ground_truth': y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true,
                        'teacher_pred_class': y_pred_teacher.numpy() if isinstance(y_pred_teacher, torch.Tensor) else y_pred_teacher,
                        'student_pred_class': y_pred_student.numpy() if isinstance(y_pred_student, torch.Tensor) else y_pred_student,
                    })
                    
                    # 添加各分类的概率值
                    if probs_teacher.ndim > 1:
                        for i in range(probs_teacher.shape[1]):
                            results_df[f'teacher_prob_class_{i}'] = probs_teacher[:, i].numpy() if isinstance(probs_teacher[:, i], torch.Tensor) else probs_teacher[:, i]
                            results_df[f'student_prob_class_{i}'] = probs_student[:, i].numpy() if isinstance(probs_student[:, i], torch.Tensor) else probs_student[:, i]
                    else:
                        results_df['teacher_prob_class_0'] = 1 - probs_teacher.numpy() if isinstance(probs_teacher, torch.Tensor) else 1 - probs_teacher
                        results_df['teacher_prob_class_1'] = probs_teacher.numpy() if isinstance(probs_teacher, torch.Tensor) else probs_teacher
                        results_df['student_prob_class_0'] = 1 - probs_student.numpy() if isinstance(probs_student, torch.Tensor) else 1 - probs_student
                        results_df['student_prob_class_1'] = probs_student.numpy() if isinstance(probs_student, torch.Tensor) else probs_student
                    
                    # 添加预测正确性标记
                    results_df['teacher_correct'] = (results_df['ground_truth'] == results_df['teacher_pred_class']).astype(int)
                    results_df['student_correct'] = (results_df['ground_truth'] == results_df['student_pred_class']).astype(int)
                
                # 确定保存路径
                if metric_path:
                    # 从metric_path获取目录和基础文件名
                    metric_dir = os.path.dirname(metric_path)
                    if not metric_dir:
                        metric_dir = '.'
                    metric_basename = os.path.basename(metric_path)
                    if metric_basename:
                        base_name = metric_basename.replace('_metrics.txt', '')
                    else:
                        base_name = 'predictions'
                    
                    # 创建predictions目录
                    predictions_dir = os.path.join(metric_dir, 'predictions')
                    os.makedirs(predictions_dir, exist_ok=True)
                    
                    # 保存文件名包含epoch信息
                    phase_suffix = f'_{self.phase_name}' if self.phase_name else ''
                    predictions_file = os.path.join(predictions_dir, f'{base_name}{phase_suffix}_epoch{epoch}.csv')
                else:
                    predictions_file = f'predictions_epoch{epoch}.csv'
                
                # 保存到CSV文件
                results_df.to_csv(predictions_file, index=False)
                print(f'预测结果已保存到: {predictions_file}')
                
            except ImportError:
                print('警告: pandas未安装，无法保存预测结果到CSV文件')
            except Exception as e:
                print(f'保存预测结果到文件时出错: {e}')

            # 计算平均验证损失
            if val_batch_count > 0:
                avg_val_loss = val_loss / val_batch_count
                avg_val_loss_itc = val_loss_itc / val_batch_count
                avg_val_loss_text_res = val_loss_text_res / val_batch_count
                avg_val_loss_image_res = val_loss_image_res / val_batch_count
                avg_val_loss_cog_res = val_loss_cog_res / val_batch_count
                avg_val_loss_cls = val_loss_cls / val_batch_count
                avg_val_loss_cls_tea = val_loss_cls_tea / val_batch_count
                avg_val_loss_cls_stu = val_loss_cls_stu / val_batch_count
                avg_val_loss_kl = val_loss_kl / val_batch_count
                avg_val_loss_feat = val_loss_feat / val_batch_count
            else:
                avg_val_loss = 0.0
                avg_val_loss_itc = 0.0
                avg_val_loss_text_res = 0.0
                avg_val_loss_image_res = 0.0
                avg_val_loss_cog_res = 0.0
                avg_val_loss_cls = 0.0
                avg_val_loss_cls_tea = 0.0
                avg_val_loss_cls_stu = 0.0
                avg_val_loss_kl = 0.0
                avg_val_loss_feat = 0.0
            
            return auc_teacher, acc_teacher, auc_student, acc_student, avg_val_loss, avg_val_loss_itc, avg_val_loss_text_res, avg_val_loss_image_res, avg_val_loss_cog_res, avg_val_loss_cls, avg_val_loss_cls_tea, avg_val_loss_cls_stu, avg_val_loss_kl, avg_val_loss_feat, f1_teacher, precision_teacher, recall_teacher, f1_student, precision_student, recall_student
        except Exception as e:
            print(f"评估过程发生错误: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def vis_tsne(
        self,
        model,
        eval_dataloader,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        img_feas = []
        txt_feas = []
        mul_feas = []
        labels = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print(eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                img_fea, txt_fea,mul_fea,label = model.tsne(data)
            # import pdb;pdb.set_trace()
            img_feas.append(img_fea)
            txt_feas.append(txt_fea)
            mul_feas.append(mul_fea)
            labels.append(label)
        img_feas = torch.cat(img_feas,dim=0).cpu()
        txt_feas = torch.cat(txt_feas,dim=0).cpu()
        mul_feas = torch.cat(mul_feas,dim=0).cpu()
        labels = torch.cat(labels,dim=0).cpu()
        

        nonlabel_indices = torch.nonzero(labels==-100).squeeze()
        img_feas = torch.index_select(img_feas, 0, torch.tensor([i for i in range(img_feas.shape[0]) if i not in nonlabel_indices]))
        txt_feas = torch.index_select(txt_feas, 0, torch.tensor([i for i in range(txt_feas.shape[0]) if i not in nonlabel_indices]))
        mul_feas = torch.index_select(mul_feas, 0, torch.tensor([i for i in range(mul_feas.shape[0]) if i not in nonlabel_indices]))
        labels = torch.index_select(labels.cpu(), 0, torch.tensor([i for i in range(labels.shape[0]) if i not in nonlabel_indices]))

        # import pdb;pdb.set_trace()

        # T-SNE
        X_train = mul_feas
        y_train = labels

        # t-SNE降维处理
        tsne = TSNE(n_components=2, verbose=1 ,random_state=42)
        result = tsne.fit_transform(X_train)

        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)

        import matplotlib.pyplot as plt

        aa = []
        bb = []
        cc = []
        for idx in range(labels.shape[0]):
            if labels[idx] == 0:
                aa.append(result[idx])
            elif labels[idx] == 1:
                bb.append(result[idx])
            else:
                cc.append(result[idx])

        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        names = ['NC', 'MCI', 'AD']
        for idx in range(3):
            if idx == 0:
                data = np.stack(aa,axis=0)
            elif idx == 1:
                data = np.stack(bb,axis=0)
            else:
                data = np.stack(cc,axis=0)
            # import pdb;pdb.set_trace()
            ax.scatter(data[:,0], data[:,1], c=colors[idx], s=10, label=names[idx],
                    alpha=0.3, cmap='viridis')

        ax.legend()
        # ax.grid(True)

        # 保存图片到指定目录
        os.makedirs('visualizations', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.savefig(f'visualizations/tsne_visualization_{timestamp}.png', dpi=200, bbox_inches='tight')
        print(f'T-SNE可视化已保存: visualizations/tsne_visualization_{timestamp}.png')
        
        # 清理plt资源
        plt.close(fig)

    def vis_tsne_bert(
        self,
        model,
        eval_dataloader,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        # img_feas = []
        txt_feas = []
        # mul_feas = []
        labels = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print(eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                txt_fea,label = model.forward_bert_tsne(data)
            txt_feas.append(txt_fea)
            labels.append(label)
        # img_feas = torch.cat(img_feas,dim=0).cpu()
        txt_feas = torch.cat(txt_feas,dim=0).cpu()
        # mul_feas = torch.cat(mul_feas,dim=0).cpu()
        labels = torch.cat(labels,dim=0).cpu()
        

        nonlabel_indices = torch.nonzero(labels==-100).squeeze()
        # img_feas = torch.index_select(img_feas, 0, torch.tensor([i for i in range(img_feas.shape[0]) if i not in nonlabel_indices]))
        txt_feas = torch.index_select(txt_feas, 0, torch.tensor([i for i in range(txt_feas.shape[0]) if i not in nonlabel_indices]))
        # mul_feas = torch.index_select(mul_feas, 0, torch.tensor([i for i in range(mul_feas.shape[0]) if i not in nonlabel_indices]))
        labels = torch.index_select(labels.cpu(), 0, torch.tensor([i for i in range(labels.shape[0]) if i not in nonlabel_indices]))

        # import pdb;pdb.set_trace()

        # T-SNE
        X_train = txt_feas
        y_train = labels

        # t-SNE降维处理
        tsne = TSNE(n_components=2, verbose=1 ,random_state=42)
        result = tsne.fit_transform(X_train)

        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)

        import matplotlib.pyplot as plt

        aa = []
        bb = []
        cc = []
        for idx in range(labels.shape[0]):
            if labels[idx] == 0:
                aa.append(result[idx])
            elif labels[idx] == 1:
                bb.append(result[idx])
            else:
                cc.append(result[idx])

        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        names = ['NC', 'MCI', 'AD']
        for idx in range(3):
            if idx == 0:
                data = np.stack(aa,axis=0)
            elif idx == 1:
                data = np.stack(bb,axis=0)
            else:
                data = np.stack(cc,axis=0)
            # import pdb;pdb.set_trace()
            ax.scatter(data[:,0], data[:,1], c=colors[idx], s=10, label=names[idx],
                    alpha=0.3, cmap='viridis')

        ax.legend()
        # ax.grid(True)

        fig.savefig('bert_4epoch.png')



        import pdb;pdb.set_trace()

    def vis_tsne_vit(
        self,
        model,
        eval_dataloader,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        img_feas = []
        # txt_feas = []
        # mul_feas = []
        labels = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print(eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                img_fea = model.forward_tsne(data['image'].cuda())
            label = data['label']
            img_feas.append(img_fea)
            labels.append(label)
        img_feas = torch.cat(img_feas,dim=0).cpu()
        # txt_feas = torch.cat(txt_feas,dim=0).cpu()
        # mul_feas = torch.cat(mul_feas,dim=0).cpu()
        labels = torch.cat(labels,dim=0).cpu()
        

        nonlabel_indices = torch.nonzero(labels==-100).squeeze()
        img_feas = torch.index_select(img_feas, 0, torch.tensor([i for i in range(img_feas.shape[0]) if i not in nonlabel_indices]))
        # txt_feas = torch.index_select(txt_feas, 0, torch.tensor([i for i in range(txt_feas.shape[0]) if i not in nonlabel_indices]))
        # mul_feas = torch.index_select(mul_feas, 0, torch.tensor([i for i in range(mul_feas.shape[0]) if i not in nonlabel_indices]))
        labels = torch.index_select(labels.cpu(), 0, torch.tensor([i for i in range(labels.shape[0]) if i not in nonlabel_indices]))

        # import pdb;pdb.set_trace()

        # T-SNE
        X_train = img_feas
        y_train = labels

        # t-SNE降维处理
        tsne = TSNE(n_components=2, verbose=1 ,random_state=42)
        result = tsne.fit_transform(X_train)

        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        result = scaler.fit_transform(result)

        import matplotlib.pyplot as plt

        aa = []
        bb = []
        cc = []
        for idx in range(labels.shape[0]):
            if labels[idx] == 0:
                aa.append(result[idx])
            elif labels[idx] == 1:
                bb.append(result[idx])
            else:
                cc.append(result[idx])

        fig, ax = plt.subplots()
        colors = ['red', 'green', 'blue']
        names = ['NC', 'MCI', 'AD']
        for idx in range(3):
            if idx == 0:
                data = np.stack(aa,axis=0)
            elif idx == 1:
                data = np.stack(bb,axis=0)
            else:
                data = np.stack(cc,axis=0)
            # import pdb;pdb.set_trace()
            ax.scatter(data[:,0], data[:,1], c=colors[idx], s=10, label=names[idx],
                    alpha=0.3, cmap='viridis')

        ax.legend()
        # ax.grid(True)

        fig.savefig('vit2.png')



        import pdb;pdb.set_trace()


    def train_bl(self,
        model,
        dataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()

        skip_scheduler = False
        for epoch in range(epochs):
            data_iterator = iter(dataloader)
            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)
                output = model(data['image'].cuda())
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(output.cpu().view(-1, 3), data['label'].view(-1))
                print('pred: ',output.argmax(-1).flatten().cpu().detach().numpy().tolist())
                print('gt  : ',data['label'].flatten().numpy().tolist())


                gts = data['label']
                preds = output
                nonlabel_indices = torch.nonzero(gts==-100).squeeze()
                # import pdb;pdb.set_trace()
                gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
                preds = torch.index_select(preds.cpu(), 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
                

                gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=3)
                preds_ont_hot = torch.nn.functional.one_hot(preds.argmax(-1), num_classes=3)
                acc = accuracy_score(preds.argmax(-1).cpu(),gts)
                auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())

                loss_value = loss / self.accumulation_steps
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}, acc: {:.4f}, auc: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value, acc, auc))
                

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
                if train_iter % 100 == 0:
                    print('save model!')
                    self._save_ckpt(model,epoch,output_path)


    def train_bert(self,
        model,
        dataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/vision_text_pretrain',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()

        skip_scheduler = False
        for epoch in range(epochs):
            data_iterator = iter(dataloader)
            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)
                # import pdb;pdb.set_trace()
                output, _, _ = model.forward_bert(data)

                loss_fct = CrossEntropyLoss()
                loss = loss_fct(output.cpu().view(-1, 3), data['label'].view(-1))
                print('pred: ',output.argmax(-1).flatten().cpu().detach().numpy().tolist())
                print('gt  : ',data['label'].flatten().numpy().tolist())


                gts = data['label']
                preds = output
                nonlabel_indices = torch.nonzero(gts==-100).squeeze()
                # import pdb;pdb.set_trace()
                gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
                preds = torch.index_select(preds.cpu(), 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
                

                gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=3)
                preds_ont_hot = torch.nn.functional.one_hot(preds.argmax(-1), num_classes=3)
                acc = accuracy_score(preds.argmax(-1).cpu(),gts)
                auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())

                loss_value = loss / self.accumulation_steps
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}, acc: {:.4f}, auc: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value, acc, auc))
                

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
            self._save_ckpt(model,epoch,output_path)


    def test_bl(
        self,
        model,
        eval_dataloader,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        gts = []
        preds = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print(eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                pred = model(data['image'].cuda())
            preds.append(pred)
            gts.append(data['label'])
        
        preds = torch.cat(preds,dim=0)
        gts = torch.cat(gts,dim=0)
        nonlabel_indices = torch.nonzero(gts==-100).squeeze()
        # import pdb;pdb.set_trace()
        gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
        preds = torch.index_select(preds.cpu(), 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
        
        gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=3)
        # import pdb;pdb.set_trace() #遇到miriad和oasis要区分一下
        # aa = preds.argmax(-1)
        # aa[aa==1]=2
        # preds_ont_hot = torch.nn.functional.one_hot(aa, num_classes=3)
        # auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())
        # acc = accuracy_score(aa.cpu(),gts.cpu())


        preds_ont_hot = torch.nn.functional.one_hot(preds.argmax(-1), num_classes=3)
        auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())
        acc = accuracy_score(preds.argmax(-1).cpu(),gts.cpu())

        print('AUC: ',auc, ' ACC: ',acc)
        # return self_attn, cross_attn

    def test_bert(
        self,
        model,
        eval_dataloader,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        gts = []
        preds = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print(eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                pred, _, _ = model.forward_bert(data)
            preds.append(pred)
            gts.append(data['label'])
        
        preds = torch.cat(preds,dim=0)
        gts = torch.cat(gts,dim=0)
        nonlabel_indices = torch.nonzero(gts==-100).squeeze()
        gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
        preds = torch.index_select(preds.cpu(), 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
        
        gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=3)

        # import pdb;pdb.set_trace() #遇到miriad和oasis要区分一下
        aa = preds.argmax(-1)
        aa[aa==1]=2
        preds_ont_hot = torch.nn.functional.one_hot(aa, num_classes=3)
        auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())
        acc = accuracy_score(aa.cpu(),gts.cpu())


        # preds_ont_hot = torch.nn.functional.one_hot(preds.argmax(-1), num_classes=3)
        # auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())
        # acc = accuracy_score(preds.argmax(-1).cpu(),gts.cpu())

        print('AUC: ',auc, ' ACC: ',acc)
        # return self_attn, cross_attn
           
    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. 
        Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts, onecycle
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'onecycle':
            # 使用PyTorch的OneCycleLR调度器，提供更灵活的学习率调整
            from torch.optim.lr_scheduler import OneCycleLR
            max_lr = optimizer.param_groups[0]['lr']
            pct_start = warmup_steps / t_total
            return OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=t_total,
                pct_start=pct_start,
                anneal_strategy='linear',
                final_div_factor=100
            )
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))
