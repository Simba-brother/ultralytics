import random

def inject_class_error(label_file_path, new_class:int):
    '''将标签文件中的class -> new_class'''
    # 先读
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    # 修改
    for i in range(len(lines)):
        line = lines[i]
        parts = line.split()
        parts[0] = f'{new_class}'
        lines[i] = " ".join(parts)
    # 写回
    with open(label_file_path, 'w') as file:
        file.writelines(lines)

def inject_box_error(label_file_path):
    '''将标签文件中的bbox进行扰动篡改'''
    # 先读
    with open(label_file_path, 'r') as file:
        lines = file.readlines()
    # 每行修改
    for i in range(len(lines)):
        cls, x_center, y_center, w, h = map(float, lines[i].split()) # 修改第i行box
        cls = int(cls)
        # 生成篡改后的标注
        x_center_err, y_center_err, w_err, h_err = perturbed_coordinates(x_center, y_center, w, h)
        lines[i] = " ".join(map(str,[cls, x_center_err, y_center_err, w_err, h_err]))
    # 写回
    with open(label_file_path, 'w') as file:
        file.writelines(lines)

def perturbed_coordinates(x_center, y_center, w, h,
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

