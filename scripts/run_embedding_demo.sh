#!/bin/bash

echo "Running CoxFormer co-expression autoencoding demo..."

EMB_NAME=coexpression

coxformer-embed \
    --emb_name $EMB_NAME \
    --epochs 20 \
    --batch_size 32

echo "Co-expression autoencoding finished."
