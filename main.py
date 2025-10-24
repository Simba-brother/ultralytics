from ultralytics import YOLO
from collections import defaultdict
import joblib
import os
import re
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd

f1_over_epoch = defaultdict(list) # dict:{int:list},class_id -> f1_list (over epoch)

def on_val_end(validator):
    metrics = validator.metrics
    box = metrics.box
    f1_list =  box.f1  # 各个类别预测的f1
    for class_id in range(len(f1_list)):
        f1_over_epoch[class_id].append(f1_list[class_id])

def traverse_directory(path):
    file_path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_path_list.append(file_path)
    return file_path_list

def change_label(label_file_path, new_label):
    # 先读
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    # 修改
    for i in range(len(lines)):
        line = lines[i]
        parts = line.split()
        parts[0] = f'{new_label}'
        lines[i] = " ".join(parts)
    # 写回
    with open(label_file_path, 'w') as file:
        file.writelines(lines)

# === 模拟人工误差 ===
def simulate_annotation_error(x_center, y_center, w, h,
                              shift_range=0.1, scale_range=0.5, noise_std=0.08):
    """
    shift_range: 中心偏移的最大比例
    scale_range: 尺度变化比例（±20%）
    noise_std:   随机噪声标准差
    """
    # 中心偏移（随机向任意方向移动）
    dx = random.uniform(-shift_range, shift_range)
    dy = random.uniform(-shift_range, shift_range)
    x_center += dx
    y_center += dy

    # 尺度误差（框变大或变小）
    scale_w = 1 + random.uniform(-scale_range, scale_range)
    scale_h = 1 + random.uniform(-scale_range, scale_range)
    w *= scale_w
    h *= scale_h

    # 添加微小噪声
    x_center += random.gauss(0, noise_std)
    y_center += random.gauss(0, noise_std)

    # 保证在[0,1]范围内
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    w = max(0.01, min(1, w))
    h = max(0.01, min(1, h))

    return x_center, y_center, w, h

def change_box(label_file_path):
    # 先读
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    # 每行修改
    for i in range(len(lines)):
        cls, x_center, y_center, w, h = map(float, lines[i].split()) # 修改第i行box
        cls = int(cls)
        # 生成篡改后的标注
        x_center_err, y_center_err, w_err, h_err = simulate_annotation_error(x_center, y_center, w, h)
        lines[i] = " ".join(map(str,[cls, x_center_err, y_center_err, w_err, h_err]))
    # 写回
    with open(label_file_path, 'w') as file:
        file.writelines(lines)

def falsify_label():
    dataset_label_dir = "datasets/african-wildlife/error_labels/train"
    print(f"准备注错的训练集目录为:{dataset_label_dir}")
    file_path_list = traverse_directory(dataset_label_dir)
    target_class_file_path_list = []
    for file_path in file_path_list:
        file_name = file_path.split("/")[-1]
        match_result = re.match(r'^1', file_name)
        if match_result:
            target_class_file_path_list.append(file_path)
    print(f"准备注错的目标设置类为:0")
    print(f"目标类图像文件数量为:{len(target_class_file_path_list)}")
    # 设置篡改的数量
    falsify_ratio = 0.05
    falsify_num = int(len(target_class_file_path_list) * falsify_ratio)
    print(f"篡改比例为:{falsify_ratio},篡改的图像数量为:{falsify_num}")
    selected_file_path_list = random.sample(target_class_file_path_list, falsify_num)
    print(f"篡改的图像文件为:")
    for id, selected_file_path in enumerate(selected_file_path_list):
        print(f"{id+1}:{selected_file_path}")
        change_label(selected_file_path,new_label=1)

def falsify_box():
    dataset_label_dir = "datasets/african-wildlife/error_labels/train"
    print(f"准备注错的训练集目录为:{dataset_label_dir}")
    file_path_list = traverse_directory(dataset_label_dir)
    # 设置篡改的数量
    falsify_ratio = 0.1
    falsify_num = int(len(file_path_list) * falsify_ratio)
    print(f"box篡改图像比例为:{falsify_ratio},box篡改图像数量为:{falsify_num}")
    selected_file_path_list = random.sample(file_path_list, falsify_num)
    print(f"篡改的图像文件为:")
    for id, selected_file_path in enumerate(selected_file_path_list):
        print(f"{id+1}:{selected_file_path}")
        # 篡改box
        change_box(selected_file_path)




def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    # Train the model
    results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640, val=True)
    # joblib.dump(f1_over_epoch,"error_f1.joblib")
    



def vis_1():
    f1_dict = joblib.load("error_f1.joblib")
    epoch_list = list(range(len(f1_dict[0])))
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, f1_dict[0], label='class_0', linewidth=2, marker='o')
    plt.plot(epoch_list, f1_dict[1], label='class_1', linewidth=2, marker='s')
    plt.plot(epoch_list, f1_dict[2], label='class_2', linewidth=2, marker='^')
    plt.plot(epoch_list, f1_dict[3], label='class_3', linewidth=2, marker='d')

    # 添加其他图表元素...
    plt.title('F1 score changes over training epochs (four-class classification), where the label of class 0 is incorrectly labeled as class 1.')
    plt.xlabel('epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("error.png")

def vis_2():
    clean_f1_dict = joblib.load("clean_f1.joblib")
    error_f1_dict = joblib.load("error_f1.joblib")
    epoch_list = list(range(len(clean_f1_dict[0])))
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, clean_f1_dict[3], label='clean_class_3', linewidth=2, marker='o', color='green')
    plt.plot(epoch_list, error_f1_dict[3], label='error_class_3', linewidth=2, marker='s', color='red')
    plt.title('Comparison of F1 scores of Class 3 under correct labeling and incorrect labeling (5% Class 0 -> Class 1)')
    plt.xlabel('epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("class3.png")

def vis_box_by_label():
    image_path = "datasets/african-wildlife/images/train/1 (64).jpg"
    label_line = "0 0.486250 0.520833 0.440000 0.515000"
    # === 解析标注 ===
    cls, x_center, y_center, w, h = map(float, label_line.split())
    # 生成篡改后的标注
    x_center_err, y_center_err, w_err, h_err = simulate_annotation_error(x_center, y_center, w, h)
    print("原始标签:", label_line)
    print(f"篡改后: 0 {x_center_err:.6f} {y_center_err:.6f} {w_err:.6f} {h_err:.6f}")
    # === 可视化对比 ===
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    def yolo_to_xyxy(xc, yc, w, h):
        xc *= W; yc *= H; w *= W; h *= H
        x1 = int(xc - w / 2); y1 = int(yc - h / 2)
        x2 = int(xc + w / 2); y2 = int(yc + h / 2)
        return x1, y1, x2, y2
    # 原始框
    x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, w, h)
    # 篡改后的框
    x1e, y1e, x2e, y2e = yolo_to_xyxy(x_center_err, y_center_err, w_err, h_err)

    # 绘制对比框：原始绿色，篡改红色
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.rectangle(img, (x1e, y1e), (x2e, y2e), (0, 0, 255), 3)
    cv2.putText(img, "original", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, "modified", (x1e, y1e - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite("compare_annotations.jpg", img)
    print("已保存对比结果：compare_annotations.jpg")

def vis_box_loss():
    clean_df = pd.read_csv("runs/detect/train/results.csv")
    box_error_df = pd.read_csv("runs/detect/train3/results.csv")
    clean_list = []
    box_error_list = []
    for index, row in clean_df.iterrows():
        clean_list.append(row["val/box_loss"])
    for index, row in box_error_df.iterrows():
        box_error_list.append(row["val/box_loss"])
    epoch_list = list(range(len(clean_list)))
    plt.figure(figsize=(12, 8))
    plt.plot(epoch_list, clean_list, label='clean_box_loss', linewidth=2, marker='o', color='green')
    plt.plot(epoch_list, box_error_list, label='error_box_loss', linewidth=2, marker='s', color='red')
    plt.title('Comparison of box_loss of correct box and incorrect box (5%)')
    plt.xlabel('epoch')
    plt.ylabel('box loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("box_loss.png")

if __name__ == "__main__":
    # main()
    # falsify_label()
    # falsify_box()
    # vis_2()
    # vis_box_by_label()
    vis_box_loss()





