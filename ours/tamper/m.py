import os
import random
import shutil
import csv
from pathlib import Path

# -----------------------------
# 配置
# -----------------------------
DATASET_DIR = "path/to/african-wildlife/train/labels"  # 原标签路径
MODIFIED_DIR = "path/to/african-wildlife/train/labels_modified"  # 备份和保存修改后的标签
CSV_FILE = "modified_labels_record.csv"  # 保存篡改记录
ERROR_RATE = 0.05  # 篡改比例
MAX_IOU_BOX_ERROR = 0.5  # box扰动IOU上限

NUM_CLASSES = 10  # 类别总数（请按实际数据集修改）

# -----------------------------
# 工具函数
# -----------------------------
def xywh_to_xyxy(x_center, y_center, w, h):
    xmin = x_center - w/2
    ymin = y_center - h/2
    xmax = x_center + w/2
    ymax = y_center + h/2
    return xmin, ymin, xmax, ymax

def xyxy_to_xywh(xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax)/2
    y_center = (ymin + ymax)/2
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
    box1Area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2Area = (box2[2]-box2[0])*(box2[3]-box2[1])
    iou = interArea / (box1Area + box2Area - interArea + 1e-8)
    return iou

def perturb_box(box, max_iou=0.5):
    xmin, ymin, xmax, ymax = box
    w = xmax - xmin
    h = ymax - ymin
    for _ in range(100):
        dx = random.uniform(-w, w)
        dy = random.uniform(-h, h)
        new_box = (xmin+dx, ymin+dy, xmax+dx, ymax+dy)
        iou = compute_iou(box, new_box)
        if 0 < iou < max_iou:
            return new_box
    return box

def move_box_far(box):
    xmin, ymin, xmax, ymax = box
    dx = random.uniform(0.5, 1.0) * random.choice([-1,1])
    dy = random.uniform(0.5, 1.0) * random.choice([-1,1])
    new_xmin = min(max(xmin+dx, 0), 1)
    new_ymin = min(max(ymin+dy, 0), 1)
    new_xmax = min(max(xmax+dx, 0), 1)
    new_ymax = min(max(ymax+dy, 0), 1)
    if new_xmax <= new_xmin: new_xmax = min(new_xmin + 0.01, 1)
    if new_ymax <= new_ymin: new_ymax = min(new_ymin + 0.01, 1)
    return new_xmin, new_ymin, new_xmax, new_ymax

# -----------------------------
# 主函数
# -----------------------------
def modify_labels():
    os.makedirs(MODIFIED_DIR, exist_ok=True)
    label_files = list(Path(DATASET_DIR).glob("*.txt"))
    num_to_modify = max(1, int(len(label_files)*ERROR_RATE))
    files_to_modify = random.sample(label_files, num_to_modify)

    csv_records = []
    csv_header = ["file_path", "error_type", "original_label"]

    # 备份所有文件
    for label_file in label_files:
        shutil.copy(label_file, MODIFIED_DIR)

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

        # 随机选择一个错误类型
        error_type = random.choice(["class", "box", "drop", "move"])

        if error_type == "class":
            new_class = class_idx
            while new_class == class_idx:
                new_class = random.randint(0, NUM_CLASSES - 1)
            lines[line_idx] = f"{new_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        elif error_type == "box":
            xmin, ymin, xmax, ymax = xywh_to_xyxy(x_center, y_center, w, h)
            xmin, ymin, xmax, ymax = perturb_box((xmin, ymin, xmax, ymax), MAX_IOU_BOX_ERROR)
            x_center, y_center, w, h = xyxy_to_xywh(xmin, ymin, xmax, ymax)
            lines[line_idx] = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        elif error_type == "drop":
            lines.pop(line_idx)

        elif error_type == "move":
            xmin, ymin, xmax, ymax = xywh_to_xyxy(x_center, y_center, w, h)
            xmin, ymin, xmax, ymax = move_box_far((xmin, ymin, xmax, ymax))
            x_center, y_center, w, h = xyxy_to_xywh(xmin, ymin, xmax, ymax)
            lines[line_idx] = f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

        # 写入修改后的label
        label_file.write_text("\n".join(lines))

        # 记录到CSV
        csv_records.append([
            str(label_file),
            error_type,
            original_label
        ])

    # 写入CSV记录文件
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_records)

    print(f"Total labels: {len(label_files)}, modified: {len(files_to_modify)}")
    print(f"CSV record saved to: {CSV_FILE}")

# -----------------------------
# 执行
# -----------------------------
if __name__ == "__main__":
    modify_labels()