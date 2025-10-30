import cv2
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def vis_classes_f1_over_epoch():
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

def vis_cleanAndError_class_f1_over_epoch():
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

def vis_box_loss():
    '''可视化clean dataset和error dataset的box loss训练曲线对比图'''
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

def vis_box_by_label():
    '''可视化篡改box'''
    image_path = "/home/mml/workspace/ultralytics/datasets/african-wildlife/images/train/4 (230).jpg"
    label_line = "0.595313 0.529054 0.460938 0.663514"
    # === 解析标注 ===
    x_center, y_center, w, h = map(float, label_line.split())
    # 生成篡改后的标注
    # x_center_err, y_center_err, w_err, h_err = perturbed_coordinates(x_center, y_center, w, h)
    x_center_err, y_center_err, w_err, h_err = 0.595313, 0.529054, 0.460938, 0.663514
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
    cv2.putText(img, "elephant", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.putText(img, "elephant", (x1e, y1e - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite("compare_annotations.jpg", img)
    print("已保存对比结果：compare_annotations.jpg")

vis_box_by_label()
