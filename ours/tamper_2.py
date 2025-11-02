import os
import random
import csv
from pathlib import Path

# -----------------------------
# 配置
# -----------------------------
DATASET_DIR = "datasets/VOC/labels/train2012" # "datasets/african-wildlife/labels/train"  # 原标签路径
CSV_FILE = "exp_results/datas/modified_labels_record_VOC_train2012.csv"  # 保存篡改记录
ERROR_RATE = 0.1  # 篡改比例
MAX_IOU_BOX_ERROR = 0.5  # box扰动IOU上限
NUM_CLASSES = 20  # 类别总数（请按实际数据集修改）

# -----------------------------
# 工具函数
# -----------------------------
def xywh_to_xyxy(x_center, y_center, w, h):
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return xmin, ymin, xmax, ymax

def xyxy_to_xywh(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return x_center, y_center, w, h

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    box1Area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2Area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))
    iou = interArea / (box1Area + box2Area - interArea + 1e-8)
    return iou

def clip_box_to_unit(xmin, ymin, xmax, ymax):
    """确保坐标归一化在 [0, 1] 范围内，并且不为零面积"""
    xmin = min(max(xmin, 0), 1)
    ymin = min(max(ymin, 0), 1)
    xmax = min(max(xmax, 0), 1)
    ymax = min(max(ymax, 0), 1)
    if xmax <= xmin:
        xmax = min(xmin + 0.01, 1)
    if ymax <= ymin:
        ymax = min(ymin + 0.01, 1)
    return xmin, ymin, xmax, ymax

def perturb_box(box, max_iou=0.5):
    """生成一个扰动后的框，保证IoU < max_iou 且归一化"""
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    for _ in range(100):
        dx = random.uniform(-w, w)
        dy = random.uniform(-h, h)
        new_box = (xmin + dx, ymin + dy, xmax + dx, ymax + dy)
        new_box = clip_box_to_unit(*new_box)
        iou = compute_iou(box, new_box)
        if 0.1 < iou < max_iou:
            return new_box
    return clip_box_to_unit(*box)

def generate_far_box(box, w, h):
    """生成一个远离原目标的新框（假框）"""
    xmin, ymin, xmax, ymax = xywh_to_xyxy(*box)
    w_new = w * random.uniform(0.8, 1.2)
    h_new = h * random.uniform(0.8, 1.2)
    for _ in range(200):
        new_xc = random.uniform(0, 1)
        new_yc = random.uniform(0, 1)
        new_xmin, new_ymin, new_xmax, new_ymax = xywh_to_xyxy(new_xc, new_yc, w_new, h_new)
        new_xmin, new_ymin, new_xmax, new_ymax = clip_box_to_unit(new_xmin, new_ymin, new_xmax, new_ymax)
        if compute_iou((xmin, ymin, xmax, ymax), (new_xmin, new_ymin, new_xmax, new_ymax)) <= 0.1:
            new_xc, new_yc, new_w, new_h = xyxy_to_xywh(new_xmin, new_ymin, new_xmax, new_ymax)
            return new_xc, new_yc, new_w, new_h
    # fallback
    return box

# -----------------------------
# 主函数
# -----------------------------
def modify_labels():
    label_files = list(Path(DATASET_DIR).glob("*.txt"))
    num_to_modify = max(1, int(len(label_files) * ERROR_RATE))
    files_to_modify = random.sample(label_files, num_to_modify)

    csv_records = []
    csv_header = ["label_file_path", "error_type", "original_label"]

    for label_file in files_to_modify:
        lines = label_file.read_text().splitlines()
        if not lines:
            continue

        # 随机选择一行标签
        line_idx = random.randint(0, len(lines) - 1)
        parts = lines[line_idx].strip().split()
        if len(parts) < 5:
            continue

        class_idx = int(parts[0])
        x_center, y_center, w, h = map(float, parts[1:])
        original_label = lines[line_idx].strip()

        # 随机选择错误类型
        error_type = random.choice(["class", "box", "drop", "redundancy"])

        # --- class error ---
        if error_type == "class":
            new_class = class_idx
            while new_class == class_idx:
                new_class = random.randint(0, NUM_CLASSES - 1)
            lines[line_idx] = f"{new_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        # --- box error ---
        elif error_type == "box":
            xmin, ymin, xmax, ymax = xywh_to_xyxy(x_center, y_center, w, h)
            xmin, ymin, xmax, ymax = perturb_box((xmin, ymin, xmax, ymax), MAX_IOU_BOX_ERROR)
            x_center, y_center, w, h = xyxy_to_xywh(xmin, ymin, xmax, ymax)
            x_center, y_center, w, h = [min(max(v, 0), 1) for v in [x_center, y_center, w, h]]
            lines[line_idx] = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        # --- drop error ---
        elif error_type == "drop":
            lines.pop(line_idx)

        # --- move error (额外添加一个多余框) ---
        elif error_type == "redundancy":
            new_xc, new_yc, new_w, new_h = generate_far_box(
                (x_center, y_center, w, h), w, h)
            # fake_class = random.randint(0, NUM_CLASSES - 1)
            fake_class = class_idx
            new_line = f"{fake_class} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
            lines.append(new_line)

        # 写入修改后的label
        label_file.write_text("\n".join(lines))

        # 记录修改信息
        csv_records.append([str(label_file), error_type, original_label])

    # 写入CSV记录
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_records)

    print(f"总标签文件: {len(label_files)}, 修改数量: {len(files_to_modify)}")
    print(f"修改记录已保存到: {CSV_FILE}")

# -----------------------------
# 执行入口
# -----------------------------
if __name__ == "__main__":
    modify_labels()