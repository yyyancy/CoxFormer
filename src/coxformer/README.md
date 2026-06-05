# CoxFormer CLI Guide
**Command-line usage for CoxFormer embedding reduction and spatial inference**

---

## Overview

This guide documents the two main CoxFormer command-line interfaces:

1. **`coxformer-embed`** - compresses transcriptome-wide co-expression vectors into lower-dimensional gene embeddings.
2. **`coxformer-impute`** - applies CoxFormer embeddings to spatial omics prediction tasks.

The examples below assume that the downloaded data are available under a local `Dataset/` folder. If your data bundle stores embeddings in a separate top-level `Embeddings/` folder, set `EMBEDDING_DIR=Embeddings` in the scripts below.

---

## Expected Data Layout

For the commands below, the project directory should contain:

```text
Dataset/
  Embeddings/
    coexpression.pkl
    CoxFormer.pkl
  Gene_expression_prediction/
    HBC1/
      cnts.tsv
      locs.tsv
      genes_train.npy
      genes_test.npy
```

`coxformer-embed` reads `coexpression.pkl` from the embedding folder and writes a reduced embedding file. `coxformer-impute` reads spatial count data from the task folder and CoxFormer gene embeddings from `CoxFormer.pkl`.

---

## Environment Variables

On macOS, scientific Python packages may load multiple OpenMP runtimes. If this happens, set the following variables before running the CLI:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export NUMBA_CACHE_DIR=/private/tmp/coxformer_numba_cache
export MPLCONFIGDIR=/private/tmp/coxformer_mpl_cache
```

These variables also provide writable cache locations for `numba` and `matplotlib`.

---

## 1. Run `coxformer-embed`

The embedding CLI reduces high-dimensional co-expression vectors and saves the reduced representation.

```bash
#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export NUMBA_CACHE_DIR=/private/tmp/coxformer_numba_cache
export MPLCONFIGDIR=/private/tmp/coxformer_mpl_cache

DATA_DIR="Dataset"
EMBEDDING_DIR="${DATA_DIR}/Embeddings"

coxformer-embed \
  --embedding_path "${EMBEDDING_DIR}" \
  --emb_name coexpression \
  --output_suffix _rd \
  --epochs 200 \
  --batch_size 32
```

Expected input:

```text
Dataset/Embeddings/coexpression.pkl
```

Expected output:

```text
Dataset/Embeddings/coexpression_rd.pkl
```

Use fewer epochs for a quick smoke test:

```bash
coxformer-embed \
  --embedding_path "Dataset/Embeddings" \
  --emb_name coexpression \
  --output_suffix _rd_test \
  --epochs 1 \
  --batch_size 32
```

---

## 2. Run `coxformer-impute`

The spatial CLI predicts held-out or unmeasured spatial molecular profiles using CoxFormer gene embeddings.

```bash
#!/usr/bin/env bash
set -euo pipefail

export KMP_DUPLICATE_LIB_OK=TRUE
export NUMBA_CACHE_DIR=/private/tmp/coxformer_numba_cache
export MPLCONFIGDIR=/private/tmp/coxformer_mpl_cache

DATA_DIR="Dataset"
EMBEDDING_DIR="${DATA_DIR}/Embeddings"
RESULT_DIR="Result"

coxformer-impute \
  --base_path "${DATA_DIR}" \
  --embedding_path "${EMBEDDING_DIR}" \
  --datasets HBC1 \
  --task Gene_expression_prediction \
  --pattern spot \
  --modality location \
  --method CoxFormer \
  --result_root "${RESULT_DIR}" \
  --epochs 200 \
  --batch_size 64
```

Expected spatial inputs:

```text
Dataset/Gene_expression_prediction/HBC1/cnts.tsv
Dataset/Gene_expression_prediction/HBC1/locs.tsv
Dataset/Gene_expression_prediction/HBC1/genes_train.npy
Dataset/Gene_expression_prediction/HBC1/genes_test.npy
Dataset/Embeddings/CoxFormer.pkl
```

Expected outputs:

```text
Result/Gene_expression_prediction/HBC1/groundtruth.csv
Result/Gene_expression_prediction/HBC1/CoxFormer-Loc_impute.csv
Result/Gene_expression_prediction/HBC1/CoxFormer-Loc_best_weights_spot_location.pt
Result/Gene_expression_prediction/HBC1/CoxFormer-Loc_loss_spot_location.pdf
```