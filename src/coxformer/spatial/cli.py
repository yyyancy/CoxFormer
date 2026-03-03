# src/coxformer/spatial/cli.py
import argparse
import os
import torch

from .utils import set_seed, ensure_dir, build_paths
from .data import (
    read_spatial_data, read_gene_embedding, read_condition,
    process_spatial_data, process_index, train_data_loader, process_embedding,
)
from .train import train_models
from .infer import (
    predict_gene_expression, predict_spot_expression,
    predict_pixel_expression, predict_cell_expression,
)


def build_argparser():
    p = argparse.ArgumentParser()

    # ===== Path configs =====
    p.add_argument("--base_path", type=str, default="./Dataset/",
                   help="Path to dataset folder")
    p.add_argument("--embedding_path", type=str, default="./Embeddings/",
                   help="Path to gene embedding folder containing <method>.pkl")
    p.add_argument("--datasets", nargs="+", default=["HBC1"],
                   help="Dataset names to run")
    p.add_argument("--task", type=str, choices=["Gene_expression_prediction", "Super_resolution_enhancement", "Gene_activity_score_prediction", "Pathological_region_detection"], default="Gene_expression_prediction")
    p.add_argument("--pattern", type=str, choices=["spot", "pixel_sim", "pixel_real"], default="spot")
    p.add_argument("--modality", nargs="+", default=["location", "image", "combine", "none"],
                   help="Modality names to run")
    p.add_argument("--result_root", type=str, default="Result")

    # ===== Method =====
    p.add_argument("--method", nargs="+", default=["CoxFormer"],
                   help="Method name(s). Will load <embedding_path>/<method>.pkl")

    # ===== Hyperparameters =====
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_non_zero", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    return p


def run(args):
    """Core runner. Can be called from CLI or notebooks."""
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for method in args.method:
        for modality in args.modality:
            model_alias = method
            if modality == "location":
                model_alias += "-Loc"
            elif modality == "image":
                model_alias += "-Img"
            elif modality == "none":
                model_alias += "-Learn"

            print(f"Method: {model_alias}")
            for dataset_idx in args.datasets:
                print(f"Dataset: {dataset_idx}")

                # paths
                data_path = os.path.join(args.base_path, args.task, dataset_idx)
                paths = build_paths(data_path, args.pattern, args.task)
                save_dir = os.path.join(args.result_root, args.task, dataset_idx)
                ensure_dir(save_dir)

                out_csv = os.path.join(save_dir, f"{model_alias}_impute.csv")
                if os.path.exists(out_csv):
                    print(f"Skip (exists): {out_csv}")
                    continue

                # load data + embedding
                spatial_data = read_spatial_data(paths["spatial"])
                emb_file = os.path.join(args.embedding_path, method + ".pkl")
                gene_embedding = read_gene_embedding(emb_file)

                # preprocess
                X_embs, y, indices_seen, indices_unseen, train_idx, all_genes = process_spatial_data(
                    spatial_data, gene_embedding, dataset_idx, paths, args.pattern,
                    random_state=args.seed, split_ratio=0.9
                )

                # condition + index
                condition = read_condition(paths, len(indices_seen), args.pattern)
                index_info = process_index(paths, condition, indices_seen, indices_unseen, train_idx, args.pattern)

                # dataloaders
                train_dataset, test_dataset, condition_dim, condition_array = train_data_loader(
                    X_embs, y, condition, index_info, all_genes, args.pattern, modality, save_dir, device
                )

                # train
                regressor, test_loader = train_models(
                    X_embs, train_dataset, test_dataset,
                    condition_dim, condition_array,
                    hidden_dim=args.hidden_dim,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    weight_non_zero=args.weight_non_zero,
                    learning_rate=args.lr,
                    device=device,
                    Modality=modality,
                    Pattern=args.pattern,
                    Method=model_alias,
                    save_dir=save_dir,
                )

                # save predictions
                if args.pattern == "spot":
                    save_dir_specific = os.path.join(save_dir, model_alias)
                    predict_gene_expression(
                        regressor, X_embs, test_loader, all_genes[indices_unseen],
                        condition_array, args.batch_size, device, save_dir_specific
                    )

                elif "pixel" in args.pattern:
                    save_dir_specific = os.path.join(save_dir, f"{model_alias}_{args.pattern}")
                    predict_spot_expression(
                        regressor, X_embs, test_loader, condition_array,
                        args.batch_size, device, save_dir_specific
                    )
                    predict_pixel_expression(
                        regressor, X_embs,
                        {**indices_unseen["test"], **indices_unseen["pred"]},
                        paths["hist_pixel"], args.batch_size * 10, device, save_dir_specific
                    )

                if "pixel_real" in args.pattern:
                    X_embs2 = process_embedding(gene_embedding)
                    predict_cell_expression(
                        regressor, X_embs2,
                        {**indices_unseen["test"], **indices_unseen["all"]},
                        paths["hist_pixel"], args.batch_size * 5, device, save_dir_specific
                    )

                print(f"Done: {dataset_idx}")


def run_impute(
    base_path="./Dataset/",
    embedding_path="./Embeddings/",
    datasets=None,
    task="Gene_expression_prediction",
    pattern="spot",
    modality=None,
    result_root="Result",
    method=None,
    hidden_dim=512,
    epochs=200,
    batch_size=64,
    lr=1e-4,
    weight_non_zero=0.5,
    seed=42,
):
    """Notebook-friendly function API."""
    parser = build_argparser()
    args = parser.parse_args([])  # defaults

    args.base_path = base_path
    args.embedding_path = embedding_path
    args.datasets = datasets if datasets is not None else ["HBC1"]
    args.task = task
    args.pattern = pattern
    args.modality = modality if modality is not None else ["location", "image", "combine", "none"]
    args.result_root = result_root
    args.method = method if method is not None else ["CoxFormer"]
    args.hidden_dim = hidden_dim
    args.epochs = epochs
    args.batch_size = batch_size
    args.lr = lr
    args.weight_non_zero = weight_non_zero
    args.seed = seed

    run(args)


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run(args)