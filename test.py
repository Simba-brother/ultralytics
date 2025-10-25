import pandas as pd

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

if __name__ == "__main__":
    pass