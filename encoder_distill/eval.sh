#!/bin/bash

python imagenet_eval.py \
        --imagenet-val "/fsx/rom1504/imagenetval/imagenetvalv1" \
        --dataset-type "webdataset" \
        --batch-size 256 \
        --workers 6 \
        --report-to "wandb" \
