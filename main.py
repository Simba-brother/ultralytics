from ultralytics import YOLO
from collections import defaultdict
import joblib
import os
import re
import random
import matplotlib.pyplot as plt

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
    print(f"篡改的图像文件为:{falsify_num}")
    for id, selected_file_path in enumerate(selected_file_path_list):
        print(f"{id+1}:{selected_file_path}")
        change_label(selected_file_path,new_label=1)

def main():
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    # model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    # Train the model
    results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640, val=True)
    joblib.dump(f1_over_epoch,"error_f1.joblib")

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

if __name__ == "__main__":
    # main()
    # falsify_label()
    vis_2()





