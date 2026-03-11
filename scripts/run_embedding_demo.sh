#!/bin/bash

echo "Running CoxFormer embedding reduction demo..."

EMB_NAME=coexpression

coxformer-embed \
    --emb_name $EMB_NAME \
    --epochs 20 \
    --batch_size 32

echo "Embedding reduction finished."