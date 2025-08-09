import os 
import pandas as pd 


# read dataset and predefined partitions (the files are available in this repository)
dataset = pd.read_csv("data/ArchiveII.csv", index_col="id")
partitions = pd.read_csv("data/ArchiveII_splits_random.csv")

for fold in range(5):
    out_path = f"working_path/{fold}/"
    os.makedirs(out_path)

    dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="train")].id].to_csv(out_path + "train.csv")
    dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="valid")].id].to_csv(out_path + "valid.csv")
    dataset.loc[partitions[(partitions.fold_number==0) & (partitions.partition=="test")].id].to_csv(out_path + "test.csv")
    
    os.system(f"sincFold -d cuda train {out_path}/train.csv --valid-file {out_path}valid.csv -o {out_path}/output/")

    os.system(f"sincFold -d cuda test {out_path}/test.csv -w {out_path}/output/weights.pmt")