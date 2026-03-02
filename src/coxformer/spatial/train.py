# ====== Basic Python Utilities ======
import os      

# ====== Numerical Computing & Data Processing ======
from tqdm import tqdm        

# ====== Deep Learning (PyTorch) ======
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ====== Visualization ======
import matplotlib.pyplot as plt     

# ====== Model ======
from .model import TransformerDecoderWithSpatialQuery, pad_embeddings_to_divisible, weighted_huber_loss


def train_models(
    X_emb,
    train_dataset,
    test_dataset,
    condition_dim,
    condition_array,
    hidden_dim=512,
    num_epochs=500,
    batch_size=128,
    learning_rate=1e-3,
    weight_non_zero=0.5,
    show_time=10,
    device="cpu",
    Modality="location",
    Pattern="spot",
    Method = "COPT",
    save_dir="Result",
):
    """
    Train TransformerDecoderWithSpatialQuery with EMA-smoothed validation early stopping.
    Automatically sets patience/warmup based on num_epochs.
    If pattern != 'spot', validation and early stopping are skipped (same as original).
    """
    modality = Modality
    pattern = Pattern

    # ==== Save path ====
    best_path = os.path.join(save_dir, f"{Method}_best_weights_{pattern}_{modality}.pt")

    # ==== Data preparation ====
    x_embeddings = torch.tensor(X_emb, dtype=torch.float32, device=device)
    x_embeddings, pad_dim = pad_embeddings_to_divisible(x_embeddings, nhead=8)
    
    # ==== Data loaders ====
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ==== Model and optimizer ====
    input_dim = int(x_embeddings.shape[1])
    print(f"input_dim:{input_dim}, condition_dim:{condition_dim}, hidden_dim:{hidden_dim}, pad_dim:{pad_dim}")

    model = TransformerDecoderWithSpatialQuery(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        Modality=modality,
        Pattern=pattern,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6, verbose=False
    )

    # ==== Loss parameters ====
    l1_coeff = 1e-3
    pos_weight = float(weight_non_zero)
    neg_weight = 1.0 - pos_weight

    # ==== Auto early-stopping parameters ====
    min_epoch_before_stop = max(20, int(0.2 * num_epochs))   # enable early stop after this epoch
    auto_patience = max(10, int(0.1 * num_epochs))          # automatic patience
    ema_beta = 0.2                                           # EMA smoothing factor
    rel_delta = 0.002                                        # relative improvement threshold (0.2%)

    # ==== Tracking variables ====
    best_state_dict = None
    best_train_smoothed = float("inf")
    ema_train = None
    train_losses = []

    # ==== Load existing model if available ====
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state)
        print(f"{best_path} exists, skip training.")
        return model, test_loader

    # ---------------- Helper Functions ---------------- #
    def compute_loss(pred, y):
        """Weighted Huber loss + L1 regularization on predictions."""
        mask_non_zero = (y > 0).float()
        loss_huber = weighted_huber_loss(
            pred, y, mask_non_zero,
            weight_zero=neg_weight,
            weight_non_zero=pos_weight
        )
        l1_out = torch.abs(pred).mean()
        return loss_huber + l1_coeff * l1_out

    def forward_one_batch(batch):
        """Unify forward pass for 'spot' and 'pixel' pattern batches."""
        if pattern != "spot":
            batch_x_idx, batch_img_idx, batch_y = batch
            bx = x_embeddings[batch_x_idx.squeeze(-1)]
            conditions = condition_array['image']
            bimg = conditions[batch_img_idx.squeeze(-1)]
            pred = model(bx, bimg)
            B = pred.shape[0]
            N = pred.shape[1]
            pred = pred.view(B, N).sum(dim=1, keepdim=True).view(1,-1)  
            y = batch_y.view(1,-1)
            bsz = bx.size(0)
            return pred, y, bsz
        else:
            bx, by = batch
            pred = model(bx, condition_array)
            bsz = bx.size(0)
            return pred, by, bsz

    def train_one_epoch():
        """Train model for one epoch."""
        model.train()
        total_loss, total = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            pred, y, bsz = forward_one_batch(batch)
            pred, y = pred.to(device), y.to(device)
            loss = compute_loss(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bsz
            total += bsz
        return total_loss / max(total, 1)

    # ---------------- Training Loop ---------------- #
    epochs_since_improve = 0  # track epochs since last improvement

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", ncols=100):
        # ---- Train ----
        train_loss = train_one_epoch()
        train_losses.append(train_loss)
    
        # EMA smoothing of train loss
        ema_train = train_loss if ema_train is None else (ema_beta * train_loss + (1 - ema_beta) * ema_train)
    
        # Log
        if epoch % show_time == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs} | Train {train_loss:.6f} | Train-EMA {ema_train:.6f}")
    
        # ---- Train-only early stopping ----
        if epoch >= min_epoch_before_stop:
            improved = (ema_train < best_train_smoothed * (1 - rel_delta))
            if improved:
                best_train_smoothed  = ema_train
                best_state_dict      = model.state_dict()
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= auto_patience:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"No >{rel_delta*100:.1f}% EMA improvement on train loss for {epochs_since_improve} epochs."
                    )
                    break
    
        # ---- Update LR once per epoch ----
        scheduler.step()

    # ---------------- Save (best or last) ---------------- #
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    torch.save(model.state_dict(), best_path)
    model.eval()

    # ---------------- Plot losses ---------------- #
    plt.figure(figsize=(6, 4))
    if len(train_losses) > 0:
        plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir + f"/{Method}_loss_{pattern}_{modality}.pdf",dpi=300)
    plt.show()

    return model, test_loader