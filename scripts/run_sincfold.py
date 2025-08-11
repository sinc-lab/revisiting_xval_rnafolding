import os 
import pandas as pd 
import json

# read dataset and predefined partitions (the files are available in this repository)
run_name = "kfold"
device = "cuda:1"

dataset = pd.read_csv("data/ArchiveII.csv", index_col="id")
if run_name == "famfold":
    partitions = pd.read_csv("data/split_famfold.csv")
elif run_name == "kfold":
    partitions = pd.read_csv("data/split_kfold.csv")
    
config = {"patience": 10}

for fold in sorted(partitions.fold_number.unique()):
    out_path = f"output_{run_name}/{fold}/"
    os.makedirs(out_path, exist_ok=True)
    json.dump(config, open(f"{out_path}/config.json", "w"))
    
    dataset.loc[partitions[(partitions.fold_number==fold) & (partitions.partition=="train")].id].to_csv(out_path + "train.csv")
    dataset.loc[partitions[(partitions.fold_number==fold) & (partitions.partition=="valid")].id].to_csv(out_path + "valid.csv")
    dataset.loc[partitions[(partitions.fold_number==fold) & (partitions.partition=="test")].id].to_csv(out_path + "test.csv")
    
    os.system(f"sincFold -d {device} -c {out_path}config.json train {out_path}/train.csv --valid-file {out_path}valid.csv -o {out_path}/output/")
    os.system(f"sincFold -d {device} pred {out_path}/test.csv -w {out_path}/output/weights.pmt -o {out_path}/output/pred.csv")