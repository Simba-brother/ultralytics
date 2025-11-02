import pandas as pd

def concat_record_df():
    train_2012_df =  pd.read_csv("exp_results/datas/modified_labels_record_VOC_train2012.csv")
    val_2012_df = pd.read_csv("exp_results/datas/modified_labels_record_VOC_val2012.csv")
    train_2007_df =  pd.read_csv("exp_results/datas/modified_labels_record_VOC_train2007.csv")
    val_2007_df =  pd.read_csv("exp_results/datas/modified_labels_record_VOC_val2007.csv")
    df_list = [train_2012_df,val_2012_df,train_2007_df,val_2007_df]
    modified_labels_record_VOC_all = pd.concat(df_list, ignore_index=True)
    modified_labels_record_VOC_all.to_csv("exp_results/datas/modified_labels_record_VOC_all.csv",index=False)
    print("modified_labels_record_VOC_all被保存在exp_results/datas/modified_labels_record_VOC_all.csv")



if __name__ == "__main__":
    concat_record_df()