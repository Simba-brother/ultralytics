import pandas as pd
import numpy as np
def test_1():
    data = {
        'sample_idx': [1,2,3,1,2,3,1,2,3],
        'epoch': [1,1,1,2,2,2,3,3,3],
        't_loss': [0.9,0.8,0.7,0.5,0.4,0.3,0.09,0.05,0.01]
    }

    df = pd.DataFrame(data)
    print(df)
    pivot_data = pd.pivot_table(df, values='t_loss', index='sample_idx', columns='epoch')
    print()

def test2():

    # 示例：原始二维数组
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    # 想要过滤掉的行索引（例如第 1 行和第 3 行）
    rows_to_remove = [1, 3]

    # 过滤掉指定行
    filtered_data = np.delete(data, rows_to_remove, axis=0)

    # 计算过滤后每列的均值
    col_means = np.mean(filtered_data, axis=0)

    print("过滤后数组：\n", filtered_data)
    print("每列均值：", col_means)

if __name__ == "__main__":
    test2()

