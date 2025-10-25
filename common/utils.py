import os
def traverse_directory(path):
    '''变异目录下所有文件'''
    file_path_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_path_list.append(file_path)
    return file_path_list