import numpy as np
import pandas as pd
import os
from scipy.stats import linregress
import topsispy as tp
import matplotlib.pyplot as plt
'''
epoch_df_list = []
epoch_idx_list = list(range(20))
for epoch_idx in epoch_idx_list:
    df = pd.read_csv(f"exp_results/datas/sample_losses/epoch_{epoch_idx}_sample_losses.csv")
    epoch_df_list.append(df)
# 合并所有epoch的数据
all_data = pd.concat(epoch_df_list, ignore_index=True)
summary_path = "exp_results/datas/sample_losses/all_epochs_sample_metrics.csv"
all_data.to_csv(summary_path, index=False)
print(f"\n所有epoch的数据已保存到: {summary_path}")
'''

'''
all_data = pd.read_csv("exp_results/datas/sample_losses/all_epochs_sample_metrics.csv")
for metric_name in ["cls_loss","box_loss","conf_sum","box_count_dif"]:
    pivot_data = all_data.pivot_table(
        index='sample_idx', # 行索引
        columns='epoch', # 列索引
        values= metric_name # 值
    )
    pivot_path = f"exp_results/datas/sample_losses/sample_{metric_name}_by_epoch.csv"
    pivot_data.to_csv(pivot_path)
    print(f"每个样本的_{metric_name}_epoch变化已保存到: {pivot_path}")
'''
'''
metric_name = "box_count_dif"
df = pd.read_csv(f"exp_results/datas/sample_losses/sample_{metric_name}_by_epoch.csv")
# 提取损失数据（假设第1列是sample_id，其余列是epoch损失）
sample_ids = df.iloc[:, 0]
metric_data = df.iloc[:, 1:].values # (nums,epochs)

data = []
for i, metrics in enumerate(metric_data):
    epochs = np.arange(len(metrics))
    # 计算均值
    mean_ = metrics.mean()
    std_ = metrics.std()
    max_ = metrics.max()
    min_ = metrics.min()
    slope_, _, _, _, _ = linregress(epochs, metrics) # 斜率
    bodong = np.std(np.diff(metrics)) # 波动性（相邻 epoch 变化量的标准差）
    item = {
        "sample_idx":i,
        "mean":mean_,
        "std":std_,
        "max":max_,
        "min":min_,
        "slop":slope_,
        "bodong":bodong,
    }
    data.append(item)
df = pd.DataFrame(data)
csv_path = f"exp_results/datas/sample_losses/{metric_name}_feature.csv"
df.to_csv(csv_path, index=False)
print(f"度量{metric_name}的feature已保存到: {csv_path}")
'''

'''
files = [
    "exp_results/datas/sample_losses/cls_loss_feature.csv", # 0 cls_loss
    "exp_results/datas/sample_losses/box_loss_feature.csv", # 1 box_loss
    "exp_results/datas/sample_losses/conf_sum_feature.csv", # 2 conf_sum
    "exp_results/datas/sample_losses/box_count_dif_feature.csv", # 3 box_count_dif
]

# 读取第一个 CSV（保留 sample_idx）
df_main = pd.read_csv(files[0])
# 依次拼接后面 3 个 CSV
for i, f in enumerate(files[1:], start=1):
    df_tmp = pd.read_csv(f)
    # 删除重复的 sample_idx 列
    df_tmp = df_tmp.drop(columns=["sample_idx"], errors='ignore')
    # 重命名列加后缀
    df_tmp = df_tmp.add_suffix(f"_{i}")
    # 合并（按行索引对齐）
    df_main = pd.concat([df_main, df_tmp], axis=1)

# 保存结果
df_main.to_csv("exp_results/datas/sample_losses/merged_feature.csv", index=False)
print("已生成合并后的 merged_feature")
'''

merged_feature_df = pd.read_csv("exp_results/datas/sample_losses/merged_feature.csv")
sample_ids = merged_feature_df.iloc[:, 0]
dataset_features = merged_feature_df.iloc[:, 1:].values # (nums,features)
'''
cls_loss:
    mean:越大越可疑 True 0
    slop:越大越可疑 True 4
    bodong:越大越可疑 True 5
box_lss:
    mean_1:越大越可疑 True 6
    slop_1:越大越可疑 True 10
    bodong_1:越大越可疑 True 11
conf_sum:
    mean_2:越小越可疑 False 12
    slop_2:越小越可疑 False 16
    bodong_2:越大越可疑 True 17
box_count_dif:
    mean_3:越大越可疑 True 18
    slop_3:越大越可疑 True 22
    bodong_3:越大越可疑 True 23
'''

feature_indices = [0,4,5,6,10,11,12,16,17,18,22,23]
dataset_subfeatures = dataset_features[:,feature_indices]
feature_signs = [1,1,1,1,1,1,-1,-1,1,1,1,1]
n_features = len(feature_signs)
weights = np.ones(n_features) / n_features
best_id, score_array = tp.topsis(dataset_subfeatures, weights, feature_signs)
# 从大到小排序并返回索引
sorted_sample_indices = np.argsort(score_array)[::-1]

sample_idx_to_imgpath = {}
epoch_0_df = pd.read_csv("exp_results/datas/sample_losses/epoch_0_sample_losses.csv")
for row_i,row in epoch_0_df.iterrows():
    sample_idx = row["sample_idx"]
    image_path = row["image_path"]
    sample_idx_to_imgpath[sample_idx] = image_path

sorted_img_path_list = []
for sample_idx in sorted_sample_indices:
    sorted_img_path_list.append(sample_idx_to_imgpath[sample_idx])

falsed_img_path_list = []
record_df = pd.read_csv("exp_results/datas/modified_labels_record.csv")
for row_i, row in record_df.iterrows():
    label_file_path = row["label_file_path"]
    # 替换目录名和扩展名
    img_file_path = label_file_path.replace("labels", "images").replace(".txt", ".jpg")
    falsed_img_path_list.append(os.path.join("/home/mml/workspace/ultralytics/",img_file_path))
print("")
def calculate_pr_metrics(list_A, list_B, thresholds=None):
    """
    计算在不同截断阈值下的 Precision 和 Recall
    
    参数:
    - list_A: 按出错概率排序的样本ID列表（越靠前越可能是错误样本）
    - list_B: 真实的错误样本ID列表
    - thresholds: 阈值列表，可以是百分比或绝对数量
    
    返回:
    - 包含不同阈值下 Precision 和 Recall 的 DataFrame
    """
    
    # 如果没有提供阈值，使用默认的百分比阈值
    if thresholds is None:
        thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,0.8,1.0]
    
    results = []
    total_error_samples = len(list_B)
    
    for threshold in thresholds:
        # 根据阈值类型确定截取数量
        if threshold < 1:  # 百分比阈值
            n = int(len(list_A) * threshold)
            threshold_type = f"{threshold*100}%"
        else:  # 绝对数量阈值
            n = min(int(threshold), len(list_A))
            threshold_type = f"{n}个"
        
        # 获取预测的错误样本
        predicted_errors = set(list_A[:n])
        
        # 计算 TP, FP, FN
        true_positives = len(predicted_errors.intersection(list_B))
        false_positives = n - true_positives
        false_negatives = total_error_samples - true_positives
        
        # 计算 Precision 和 Recall
        precision = true_positives / n if n > 0 else 0
        recall = true_positives / total_error_samples if total_error_samples > 0 else 0
        
        # 计算 F1-score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold_type,
            'n_samples': n,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        })
    
    return pd.DataFrame(results)

def analyse():
    results_df = calculate_pr_metrics(sorted_img_path_list,falsed_img_path_list)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['threshold'], results_df['precision'], 'b-', linewidth=2, label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], 'r-', linewidth=2, label='Recall')
    plt.xlabel('cut off')
    plt.ylabel('score')
    plt.title('Precision and Recall over cut off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("exp_results/datas/PR_1.png")

def compute_apfd(list_A, list_B):
    """
    list_A: set/list, 真实错误图像路径
    list_B: list, 按可疑度排序的图像路径
    """
    n = len(list_B)
    list_A_set = set(list_A)
    TF_positions = []

    # 遍历 list_B 找到真实错误的位置
    for idx, img in enumerate(list_B, start=1):  # 从1开始计数
        if img in list_A_set:
            TF_positions.append(idx)

    m = len(list_A)
    if m == 0:
        return 0.0  # 防止除零

    apfd = 1 - sum(TF_positions) / (n * m) + 1 / (2 * n)
    return apfd

analyse()

apfd = compute_apfd(falsed_img_path_list,sorted_img_path_list)
print("")
















