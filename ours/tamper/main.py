import re
import random
from common.utils import traverse_directory
from ours.tamper.inject_error import inject_class_error, inject_box_error

def tamper_class(dataset_label_dir:str):
    print(f"准备注错的训练集目录为:{dataset_label_dir}")
    target_class = 0
    print(f"准备注错的目标设置类为:{target_class}")
    file_path_list = traverse_directory(dataset_label_dir)
    target_class_file_path_list = []
    pattern = "^{0}".format(target_class+1)
    compiled_pattern = re.compile(pattern)
    for file_path in file_path_list:
        file_name = file_path.split("/")[-1]
        match_result = compiled_pattern.match(file_name) # 匹配失败返回None
        if match_result:
            target_class_file_path_list.append(file_path)
    print(f"目标类图像文件数量为:{len(target_class_file_path_list)}")
    # 设置篡改的数量
    falsify_ratio = 0.05
    falsify_num = int(len(target_class_file_path_list) * falsify_ratio)
    print(f"篡改比例为:{falsify_ratio},篡改的图像数量为:{falsify_num}")
    selected_file_path_list = random.sample(target_class_file_path_list, falsify_num)
    print(f"篡改的图像文件为:")
    for id, selected_file_path in enumerate(selected_file_path_list):
        print(f"{id+1}:{selected_file_path}")
        inject_class_error(selected_file_path,new_class=1)

def tamper_box(dataset_label_dir:str):
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
        inject_box_error(selected_file_path)


if __name__ == "__main__":
    dataset_label_dir = "datasets/african-wildlife/labels/train"
    tamper_class(dataset_label_dir)