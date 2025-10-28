import re
import random
import pandas as pd
from common.utils import traverse_directory
from ours.tamper.inject_error import inject_class_error, inject_box_error

def tamper_class(dataset_label_dir:str, ):
    print(f"训练集标签目录:{dataset_label_dir}")
    file_path_list = traverse_directory(dataset_label_dir)
    print(f"总共有标签文件数量:{file_path_list}")
    # 篡改率2%
    falsify_ratio = 0.02
    falsify_num = max(int(len(file_path_list) * falsify_ratio),1) # 最少1个
    print(f"class注错率:{falsify_ratio},数量为:{falsify_num}")
    selected_file_path_list = random.sample(file_path_list, falsify_num)
    print(f"篡改class的标签文件为:")
    for id, selected_file_path in enumerate(selected_file_path_list):
        print(f"{id+1}:{selected_file_path}")
        o_class_indices,n_class_indices = inject_class_error(selected_file_path)
    # 记录篡改的文件到csv中
    record_data = []
    for id, selected_file_path in enumerate(selected_file_path_list):
        img_path = selected_file_path.replace("labels","images")
        img_path = img_path.replace("txt","jpg")
        item = {
            "img_path":img_path, 
            "label_path":selected_file_path,
            "o_class_indices":str(o_class_indices),
            "n_class_indices":str(n_class_indices)}
        record_data.append(item)
    record_data_df = pd.DataFrame(record_data)
    save_file_path = "exp_results/datas/class_label_falsify_record.csv"
    record_data_df.to_csv(save_file_path, index=False)
    print(f"篡改记录保存在:{save_file_path}")

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