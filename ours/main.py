from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
import os
from scipy.stats import linregress
# 加载训练数据集
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset

class PerSampleLossTracker:
    """追踪每个训练样本的loss"""

    def __init__(self, model, save_dir='exp_results/datas/sample_training_metrics_2'):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.epoch_losses = [] # 存储每个epoch的df
    
    def calculate_sample_losses(self, data_yaml):
        """计算训练集每个样本的loss"""
        epoch_idx = self.model.trainer.epoch
        print(f"\n计算 Epoch {epoch_idx} 的样本训练信息...")
        # 设置模型为评估模式
        self.model.model.eval()
        # 检查数据集配置
        data_dict = check_det_dataset(data_yaml)
        # 构建训练集的dataloader (batch_size=1 以获取每个样本)
        # 根据yaml把训练集给构建出来
        dataset = self.model.trainer.build_dataset(
            data_dict["train"], # self.model.trainer.args.data, # data_dict, 
            mode='val',
            batch=1
        )
        # 基于数据集，构建出数据加载器，因为需要样本级别的loss，所以数据加载器的batch=1,为了好追溯shuffle=False
        dataloader = build_dataloader(
            dataset=dataset,
            batch=1,  # 每次只处理一个样本
            workers=0,
            shuffle=False
        )
        # 用于保存所有训练样本的loss info
        sample_losses = []
        epoch_pt_path = self.model.trainer.save_dir / "weights" / f"epoch{epoch_idx}.pt"
        temp_yolo = YOLO(epoch_pt_path)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # 将数据移到正确的设备
                batch = self.model.trainer.preprocess_batch(batch)
                # 前向传播
                preds = self.model.model(batch['img'])
                # 计算loss
                loss, loss_items = self.model.model.criterion(preds, batch)
                box_loss = loss_items[0].item() # 度量预测框与真实框的重叠程度 box_loss = 1 - IOU(Box_pred, Box_gt)
                cls_loss = loss_items[1].item()
                dfl_loss = loss_items[2].item() # 框子微调
                total_loss = box_loss + cls_loss + dfl_loss
                # 计算confi
                im_file = batch["im_file"][0]
                results = temp_yolo(im_file)
                result = results[0]
                # 获得样本的confidence
                conf_sum = result.boxes.conf.sum().item()
                # 获得 predicted box count 与 ground truth box count 的差值
                gt_box_count = batch["cls"].shape[0]
                predicted_box_count = result.boxes.conf.numel()
                box_count_dif = abs(gt_box_count-predicted_box_count)
                # 记录该样本的training data
                sample_info = {
                    'epoch': epoch_idx,
                    'sample_idx': batch_idx,
                    'image_path': str(batch['im_file'][0]) if 'im_file' in batch else f"sample_{batch_idx}",
                    'total_loss': total_loss,
                    'box_loss': box_loss, # box损失,IOU相关
                    'cls_loss': cls_loss, # 类损失
                    'dfl_loss': dfl_loss,  # dfl 框子回归损失,框子微调相关
                    "conf_sum":conf_sum, # 置信度
                    "box_count_dif":box_count_dif # box 个数
                }
                sample_losses.append(sample_info)
                
                if (batch_idx + 1) % 100 == 0: # 每完成100个batch(sample)就会打印一下
                    print(f"已处理 {batch_idx + 1}/{len(dataloader)} 个样本")
        # 保存该epoch的结果
        df = pd.DataFrame(sample_losses)
        csv_path = self.save_dir / f'epoch_{epoch_idx}_sample_losses.csv'
        df.to_csv(csv_path, index=False)
        print(f"已保存到: {csv_path}")

        '''
        # 计算损失
    
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self.model.trainer.preprocess_batch(batch)
                # 模型的前向传播
                preds = self.model.model(batch['img'])
                # 计算loss
                loss, loss_items = self.model.model.criterion(preds, batch) # self.model.loss(batch)
                # YOLO call 推理目的是获得confidence信息
                results = self.model(batch["img"])
                result = results[0]
                
                # 获得样本的loss data
                box_loss = loss_items[0].item() # 度量预测框与真实框的重叠程度 box_loss = 1 - IOU(Box_pred, Box_gt)
                cls_loss = loss_items[1].item()
                dfl_loss = loss_items[2].item() # 框子微调
                total_loss = box_loss + cls_loss + dfl_loss
                
                # 获得样本的confidence
                conf_sum = result.boxes.conf.sum().item()
                # 获得 predicted box count 与 ground truth box count 的差值
                gt_box_count = result.boxes.cls.numel()
                predicted_box_count = result.boxes.conf.numel()
                box_count_dif = abs(gt_box_count-predicted_box_count)
                

                # 记录该样本的loss
                sample_info = {
                    'epoch': epoch,
                    'sample_idx': batch_idx,
                    'image_path': str(batch['im_file'][0]) if 'im_file' in batch else f"sample_{batch_idx}",
                    'total_loss': total_loss,
                    'box_loss': box_loss, # box损失,IOU相关
                    'cls_loss': cls_loss, # 类损失
                    'dfl_loss': dfl_loss,  # dfl 框子回归损失,框子微调相关
                    # "conf_sum":conf_sum, # 置信度
                    # "box_count_dif":box_count_dif # box 个数
                }
                sample_losses.append(sample_info)
                
                if (batch_idx + 1) % 100 == 0: # 每完成100个batch(sample)就会打印一下
                    print(f"已处理 {batch_idx + 1}/{len(dataloader)} 个样本")
        
        # 保存该epoch的结果
        df = pd.DataFrame(sample_losses)
        csv_path = self.save_dir / f'epoch_{epoch}_sample_losses.csv'
        df.to_csv(csv_path, index=False)
        print(f"已保存到: {csv_path}")
        
        # 统计信息
        print(f"\nEpoch {epoch} 统计:")
        print(f"平均total_loss: {df['total_loss'].mean():.4f}")
        print(f"平均box_loss: {df['box_loss'].mean():.4f}")
        print(f"平均cls_loss: {df['cls_loss'].mean():.4f}")
        print(f"平均dfl_loss: {df['dfl_loss'].mean():.4f}")
        print(f"最大Total loss样本索引: {df['total_loss'].idxmax()}, loss: {df['total_loss'].max():.4f}")
        print(f"最小Total loss样本索引: {df['total_loss'].idxmin()}, loss: {df['total_loss'].min():.4f}")
        
        self.epoch_losses.append(df) # 每个epoch组织成一个df
        self.model.model.train()
        return df
    '''
    
    def save_summary(self):
        """保存所有epoch的汇总"""
        if not self.epoch_losses:
            return
        # 合并所有epoch的数据
        all_data = pd.concat(self.epoch_losses, ignore_index=True)
        summary_path = self.save_dir / 'all_epochs_sample_losses.csv'
        all_data.to_csv(summary_path, index=False)
        print(f"\n所有epoch的数据已保存到: {summary_path}")
        
        # 生成每个样本的loss变化
        pivot_data = all_data.pivot_table(
            index='sample_idx', # 行索引
            columns='epoch', # 列索引
            values='cls_loss' # 值
        )
        pivot_path = self.save_dir / 'sample_loss_by_epoch.csv'
        pivot_data.to_csv(pivot_path)
        print(f"每个样本的epoch变化已保存到: {pivot_path}")

def train_with_sample_loss_tracking(model_path='yolo11n.pt', 
                                   data_yaml='coco128.yaml',
                                   epochs=10,
                                   **train_kwargs):
    """带样本loss追踪的训练"""
    
    # 初始化模型
    model = YOLO(model_path)
    # 创建loss追踪器
    loss_tracker = PerSampleLossTracker(model)
    # 定义回调函数：每个checkpoint后计算训练样本loss和confi
    def on_model_save(trainer):
        loss_tracker.calculate_sample_losses(data_yaml)
    # 添加回调
    model.add_callback("on_model_save", on_model_save)
    # 开始训练
    print("开始训练...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        **train_kwargs
    )
    # 保存汇总
    loss_tracker.save_summary()
    return model, loss_tracker


def rank_samples_by_loss_decline(csv_file_path):
    """
    基于损失下降梯度对样本进行排序
    
    参数:
    - csv_file_path: 包含损失数据的CSV文件路径
    """
    
    # 读取数据
    df = pd.read_csv(csv_file_path)
    
    # 提取损失数据（假设第1列是sample_id，其余列是epoch损失）
    sample_ids = df.iloc[:, 0]
    loss_data = df.iloc[:, 1:].values
    
    # 计算每个样本的损失下降梯度
    slopes = []
    
    for i, losses in enumerate(loss_data):
        # 计算损失下降趋势（线性回归斜率）
        epochs = np.arange(len(losses))
        slope, _, _, _, _ = linregress(epochs, losses)
        slopes.append(slope)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'sample_idx': sample_ids,
        'loss_decline_slope': slopes
    })
    
    # 按损失下降梯度排序（斜率越小，下降越快）
    # 注意：斜率负值越大表示下降越快，正值表示损失上升
    result_df = result_df.sort_values('loss_decline_slope', ascending=False)
    
    # 重置索引并添加排名
    result_df = result_df.reset_index(drop=True)
    result_df['rank'] = result_df.index + 1
    
    print("基于损失下降梯度的样本排序（前20个最可疑的样本）:")
    print("排名越高（rank越小）的样本损失下降越慢，越可能是错误标注")
    print(result_df.head(5))
    epoch_0_df = pd.read_csv("exp_results/datas/sample_losses/epoch_0_sample_losses.csv")
    # 基于 sample_id 合并两个 DataFrame，只添加 image_path 列到 df2
    rank_df = pd.merge(
        result_df, 
        epoch_0_df[['sample_idx', 'image_path']], 
        on='sample_idx', 
        how='left'
    )
    rank_df.to_csv("exp_results/datas/rank.csv",index=False)
    print("rank csv保存在:exp_results/datas/rank.csv")
    return rank_df






def test(epoch,save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    # 初始化模型
    yolo = YOLO(f"/home/mml/workspace/ultralytics/runs/detect/train21/weights/epoch{epoch}.pt")
    data_yaml = "african-wildlife.yaml"
    data_dict = check_det_dataset(data_yaml)
    # 构建训练集的dataloader (batch_size=1 以获取每个样本)
    # 根据yaml把训练集给构建出来
    dataset = yolo.trainer.build_dataset(
        data_dict["train"], # self.model.trainer.args.data, # data_dict, 
        mode='val',
        batch=1
    )
    # 基于数据集，构建出数据加载器，因为需要样本级别的loss，所以数据加载器的batch=1,为了好追溯shuffle=False
    dataloader = build_dataloader(
        dataset=dataset,
        batch=1,  # 每次只处理一个样本
        workers=0,
        shuffle=False
    )

    # 用于保存所有训练样本的loss info
    sample_losses = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = yolo.trainer.preprocess_batch(batch)
            # 获得损失
            loss,loss_items = yolo.loss(batch['img'])
            # 获得预测结果
            results = yolo(batch["img"])
            result = results[0]
            # 获得样本的loss data
            box_loss = loss_items[0].item() # 度量预测框与真实框的重叠程度 box_loss = 1 - IOU(Box_pred, Box_gt)
            cls_loss = loss_items[1].item()
            dfl_loss = loss_items[2].item() # 框子微调
            total_loss = box_loss + cls_loss + dfl_loss
            
            # 获得样本的confidence
            conf_sum = result.boxes.conf.sum().item()
            # 获得 predicted box count 与 ground truth box count 的差值
            gt_box_count = result.boxes.cls.numel()
            predicted_box_count = result.boxes.conf.numel()
            box_count_dif = abs(gt_box_count-predicted_box_count)
            # 记录该样本的loss
            sample_info = {
                'epoch': epoch,
                'sample_idx': batch_idx,
                'image_path': str(batch['im_file'][0]) if 'im_file' in batch else f"sample_{batch_idx}",
                'total_loss': total_loss,
                'box_loss': box_loss, # box损失,IOU相关
                'cls_loss': cls_loss, # 类损失
                'dfl_loss': dfl_loss,  # dfl 框子回归损失,框子微调相关
                # "conf_sum":conf_sum, # 置信度
                # "box_count_dif":box_count_dif # box 个数
            }
            sample_losses.append(sample_info)
            
            if (batch_idx + 1) % 100 == 0: # 每完成100个batch(sample)就会打印一下
                print(f"已处理 {batch_idx + 1}/{len(dataloader)} 个样本")
    
    # 保存该epoch的结果
    df = pd.DataFrame(sample_losses)
    csv_path = save_dir / f'epoch_{epoch}_sample_losses.csv'
    df.to_csv(csv_path, index=False)
    print(f"已保存到: {csv_path}")


# 使用示例
if __name__ == "__main__":
    # 方式1: 使用自动追踪功能训练
    
    model = train_with_sample_loss_tracking(
        model_path='yolo11n.pt',
        data_yaml='african-wildlife.yaml',
        epochs=20,
        imgsz=640,
        batch=32,
        device=0,
        save_period=1,
    )


    # 分析样本在训练过程中指标趋势
    # rank_samples_by_loss_decline("exp_results/datas/sample_losses/sample_loss_by_epoch.csv")
    # analyse()
    # 方式2: 对已训练的模型，单独计算某个epoch的样本loss
    """
    model = YOLO('runs/detect/train/weights/best.pt')
    tracker = PerSampleLossTracker(model)
    
    # 需要先初始化trainer
    model.train(data='coco128.yaml', epochs=1, resume=True)
    
    # 计算样本loss
    sample_losses = tracker.calculate_sample_losses('coco128.yaml', epoch=0)
    """
    
    # 读取并分析结果
    """
    import pandas as pd
    
    # 读取某个epoch的样本loss
    df = pd.read_csv('sample_losses/epoch_0_sample_losses.csv')
    
    # 找出loss最高的前10个样本
    top_10_hardest = df.nlargest(10, 'total_loss')
    print("\nLoss最高的10个样本:")
    print(top_10_hardest[['sample_idx', 'image_path', 'total_loss']])
    
    # 读取所有epoch的数据
    all_data = pd.read_csv('sample_losses/all_epochs_sample_losses.csv')
    
    # 分析某个样本在不同epoch的loss变化
    sample_0_losses = all_data[all_data['sample_idx'] == 0]
    print("\n样本0在各个epoch的loss:")
    print(sample_0_losses[['epoch', 'total_loss']])
    """