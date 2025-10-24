from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import numpy as np
# 加载训练数据集
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset

class PerSampleLossTracker:
    """追踪每个训练样本的loss"""
    
    def __init__(self, model, save_dir='sample_losses'):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.epoch_losses = [] # 存储每个epoch的df
    
    def calculate_sample_losses(self, data_yaml, epoch):
        """计算训练集每个样本的loss"""
        print(f"\n计算 Epoch {epoch} 的样本loss...")
        
        # 设置模型为评估模式
        self.model.model.eval()

        # 检查数据集配置
        data_dict = check_det_dataset(data_yaml)
        
        # 构建训练集的dataloader (batch_size=1 以获取每个样本)
        # 根据yaml把训练集给构建出来
        dataset = self.model.trainer.build_dataset(
            data_dict["train"], # self.model.trainer.args.data, # data_dict, 
            mode='train',
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
                # 将数据移到正确的设备
                batch = self.model.trainer.preprocess_batch(batch)
                
                # 前向传播
                preds = self.model.model(batch['img'])
                
                # 计算loss
                loss, loss_items = self.model.model.criterion(preds, batch)
                box_loss = loss_items[0].item()
                cls_loss = loss_items[1].item()
                dfl_loss = loss_items[2].item()
                total_loss = 7.5*box_loss+0.5*cls_loss+1.5*dfl_loss
                # 记录该样本的loss
                sample_info = {
                    'epoch': epoch,
                    'sample_idx': batch_idx,
                    'image_path': str(batch['im_file'][0]) if 'im_file' in batch else f"sample_{batch_idx}",
                    'total_loss': total_loss,
                    'box_loss': box_loss,
                    'cls_loss': cls_loss,
                    'dfl_loss': dfl_loss
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
        
        self.epoch_losses.append(df)
        
        return df
    
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
            values='total_loss' # 值
        )
        pivot_path = self.save_dir / 'sample_loss_by_epoch.csv'
        pivot_data.to_csv(pivot_path)
        print(f"每个样本的epoch变化已保存到: {pivot_path}")


def train_with_sample_loss_tracking(model_path='yolov8n.pt', 
                                   data_yaml='coco128.yaml',
                                   epochs=10,
                                   **train_kwargs):
    """带样本loss追踪的训练"""
    
    # 初始化模型
    model = YOLO(model_path)
    
    # 创建loss追踪器
    loss_tracker = PerSampleLossTracker(model)
    
    # 定义回调函数：每个epoch结束后计算训练样本loss
    def on_train_epoch_end(trainer):
        epoch = trainer.epoch
        # 计算并保存当前epoch每个样本的loss
        loss_tracker.calculate_sample_losses(data_yaml, epoch)
    
    # 添加回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    
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


# 使用示例
if __name__ == "__main__":
    # 方式1: 使用自动追踪功能训练
    model, tracker = train_with_sample_loss_tracking(
        model_path='yolo11n.pt',
        data_yaml='african-wildlife.yaml',
        epochs=3,
        imgsz=640,
        batch=16
    )
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