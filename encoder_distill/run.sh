#!/bin/bash

python3.8 train.py \
        --train-data "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231348}.tar -" \
        --train-num-samples 2000000000 \
        --val-data "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{231349..231349}.tar -" \
        --val-num-samples 10000 \
        --imagenet-val "/fsx/rom1504/imagenetval/imagenetvalv1" \
        --dataset-type "webdataset" \
        --batch-size 128 \
        --lr 2e-4 \
        --beta1 0.9 \
        --beta2 0.98 \
        --eps 1e-6 \
        --wd 0.0 \
        --workers 6 \
        --steps 100000 \
        --warmup 1000 \
        --modality "image" \
        --save-frequency 1000 \
        --val-frequency 100 \
        --report-to "wandb" \
        --name "H 2048 bs 2e-4 lr" \
