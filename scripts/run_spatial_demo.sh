#!/bin/bash

echo "Running CoxFormer spatial molecular prediction demo..."

DATASET=HBC1

coxformer-impute \
    --datasets $DATASET \
    --method CoxFormer \
    --epochs 20 \
    --batch_size 32

echo "Spatial molecular prediction finished."
