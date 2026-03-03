# src/coxformer/embedding/cli.py
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CoexpressDataset
from .model import CoxformerAE


def build_argparser():
    p = argparse.ArgumentParser()

    # ===== Path configs =====
    p.add_argument(
        "--embedding_path",
        type=str,
        default="/home/wangshu/scratch/COPT/Embeddings/",
        help="Path to embedding folder containing <emb_name>.pkl",
    )
    p.add_argument(
        "--emb_name",
        type=str,
        default="coexpression",
        help="Embedding file stem. Will load <embedding_path>/<emb_name>.pkl",
    )
    p.add_argument(
        "--output_suffix",
        type=str,
        default="_rd",
        help="Suffix for output file. Output will be <emb_name><suffix>.pkl",
    )

    # ===== Chunking / model configs =====
    p.add_argument("--seq_length", type=int, default=512)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512)

    # ===== Training configs =====
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)

    # ===== Optim configs =====
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--t_max", type=int, default=50)
    p.add_argument("--eta_min", type=float, default=1e-6)

    # ===== Loader configs =====
    p.add_argument("--num_workers", type=int, default=0)

    return p


def run(args):
    """Core runner. Can be called from CLI or notebooks."""
    emb_name = args.emb_name
    file_path = os.path.join(args.embedding_path, f"{emb_name}.pkl")

    with open(file_path, "rb") as file:
        df = pickle.load(file)

    # Extract the Coexpress embeddings and convert them to a NumPy array
    coexpress_embeddings = np.array(df["Embedding"].tolist())

    # Ensure the embeddings are of type float32
    coexpress_embeddings = coexpress_embeddings.astype(np.float32)

    # Check for NaNs or infinite values
    if np.isnan(coexpress_embeddings).any() or np.isinf(coexpress_embeddings).any():
        print("Data contains NaNs or infinite values. Please clean your data.")
        return

    # Get input_dim and save original_input_dim
    input_dim = coexpress_embeddings.shape[1]
    original_input_dim = input_dim
    print(f"Original input_dim: {input_dim}")

    # Set seq_length and embedding_dim
    seq_length = args.seq_length
    embedding_dim = args.embedding_dim

    # Ensure input_dim is divisible by seq_length
    if input_dim % seq_length != 0:
        padding_size = seq_length - (input_dim % seq_length)
        coexpress_embeddings = np.pad(
            coexpress_embeddings, ((0, 0), (0, padding_size)), mode="constant"
        )
        input_dim = coexpress_embeddings.shape[1]
        print(f"Adjusted input_dim after padding: {input_dim}")

    # Now compute chunk_size
    chunk_size = input_dim // seq_length
    print(f"Chunk size: {chunk_size}")

    dataset = CoexpressDataset(coexpress_embeddings)
    batch_size = args.batch_size
    device_tmp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = True if device_tmp.type == "cuda" else False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )

    # Initialize the model
    hidden_dim = args.hidden_dim
    model = CoxformerAE(
        input_dim=input_dim,
        original_input_dim=original_input_dim,
        hidden_dim=hidden_dim,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DataParallel to utilize multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.t_max,
        eta_min=args.eta_min,
    )

    # Training loop
    num_epochs = args.epochs
    patience = args.patience
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", ncols=100):
        model.train()
        running_loss = 0.0

        for data in dataloader:
            data = data.to(device, non_blocking=True)

            # Forward pass
            outputs, _ = model(data)
            loss = criterion(outputs, data)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}")

        scheduler.step()

        # EarlyStopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement in the last {patience} epochs)."
                )
                break

    # Extract embeddings from the encoder
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for data in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers):
            data = data.to(device, non_blocking=True)
            outputs, encoded = model(data)
            all_embeddings.append(encoded.detach().cpu().numpy())

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)

    # Add the new embeddings to the DataFrame
    df["Embedding"] = list(all_embeddings)

    # Save the updated DataFrame
    output_file = os.path.join(args.embedding_path, f"{emb_name}{args.output_suffix}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(df, f)

    print(f"Embeddings reduced and saved to {output_file}")


def run_reduce(
    embedding_path="/home/wangshu/scratch/COPT/Embeddings/",
    emb_name="coexpression",
    output_suffix="_rd",
    seq_length=512,
    embedding_dim=64,
    hidden_dim=512,
    batch_size=32,
    epochs=200,
    patience=50,
    lr=1e-4,
    t_max=50,
    eta_min=1e-6,
    num_workers=0,
):
    """Notebook-friendly function API."""
    parser = build_argparser()
    args = parser.parse_args([])  # defaults

    args.embedding_path = embedding_path
    args.emb_name = emb_name
    args.output_suffix = output_suffix

    args.seq_length = seq_length
    args.embedding_dim = embedding_dim
    args.hidden_dim = hidden_dim

    args.batch_size = batch_size
    args.epochs = epochs
    args.patience = patience

    args.lr = lr
    args.t_max = t_max
    args.eta_min = eta_min

    args.num_workers = num_workers

    run(args)


def main():
    parser = build_argparser()
    args = parser.parse_args()
    run(args)