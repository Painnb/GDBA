#!/bin/bash
# 定义数据集和扰动率的组合
datasets=("cora" "citeseer" "blogcatalog" "uai" "acm" "flickr")
ptb_rates=(0 0.05 0.1 0.2)
# 遍历每种数据集和扰动率组合
for dataset in "${datasets[@]}"; do
  for ptb_rate in "${ptb_rates[@]}"; do
    echo "Running experiment with dataset=${dataset}, ptb_rate=${ptb_rate}"
    # 使用 nohup 运行 Python 程序，并将结果记录到 results.csv 文件中
    python main_dice.py --dataset "$dataset" --ptb_rate "$ptb_rate" &
    wait
  done
done

