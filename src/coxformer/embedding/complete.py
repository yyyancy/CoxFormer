import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import time
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
from tensorboardX import SummaryWriter

from coxformer.embedding.data import CoxformerDataset, split_labeled_edges_indices
from coxformer.embedding.model import CoxformerNet
from coxformer.embedding.train import CoxformerTrainer
from coxformer.embedding.infer import infer_coexpression, print_analysis

def main(args):
    for subdir in [f'out/{args.project_name}', f'runs/{args.project_name}']:
        if not os.path.exists(subdir):
            os.makedirs(subdir, exist_ok=True)
            print(f'Created directory: {subdir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'GPU total memory: {gpu_memory:.2f} GB')

    print("Loading data...")
    dataset = CoxformerDataset(args.coexpression_file, args.correlation_file, args.embedding_file)

    data, gene_to_idx, all_nodes_genes, common_genes, target_min, target_max, edge_corr_min, edge_corr_max = dataset.create_enhanced_graph_data(args.top_k_edges)

    train_indices, val_indices, test_indices = split_labeled_edges_indices(data)
    data = data.to(device)

    print('Data split summary:')
    print(f'Total edges: {data.edge_index.shape[1]}')
    print(f'Labeled edges: {data.label_mask.sum().item()}')
    print(f'Unlabeled edges:{(~data.label_mask).sum().item()}')
    print(f'Train edges: {len(train_indices)}')
    print(f'Val edges: {len(val_indices)}')
    print(f'Test edges: {len(test_indices)}')

    input_dim = data.x.shape[1]
    edge_dim = args.edge_dim
    hidden_dims = args.hidden_dims

    print('Model configuration:')
    print(f'Node feature dim: {input_dim}')
    print(f'Edge feature dim: {edge_dim}')
    print(f'Hidden dims: {hidden_dims}')

    model = CoxformerNet(input_dim, hidden_dims, edge_dim=edge_dim, dropout=args.dropout, use_edge_features=True)
    model = model.to(device)

    trainer = CoxformerTrainer(model, device, num_neighbors=args.num_neighbors, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    writer = SummaryWriter(f'runs/{args.project_name}/training')
    print(f'TensorBoard log dir: runs/{args.project_name}/training')

    train_losses = []
    training_duration = 0.0

    if not args.only_eval:
        print("Starting Training...")
        num_epochs = args.num_epochs
        training_start_time = time.time()

        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(data, optimizer, criterion, train_indices)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.flush()

            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}')
            train_losses.append(train_loss)
            scheduler.step()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.save(model.state_dict(), f'out/{args.project_name}/best_model.pth')
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        print(f'Training time: {training_duration:.2f} s')
        print(f'Model saved to: out/{args.project_name}/best_model.pth')
    else:
        model_path = f'out/{args.project_name}/best_model.pth'
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}. Cannot run --only_eval without a trained model.")

    print("Running test evaluation...")
    # 确保加载的是最新保存的权重（如果是刚训练完）
    if not args.only_eval:
        model.load_state_dict(torch.load(f'out/{args.project_name}/best_model.pth'))
        
    test_loss, test_mse, test_mae, test_r2, test_pred, test_true = trainer.evaluate(data, criterion, test_indices)

    print('Test results:')
    print(f'MSE: {test_mse:.4f}')
    print(f'MAE: {test_mae:.4f}')
    print(f'R2 Score: {test_r2:.4f}')

    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('R2/test',   test_r2,   0)
    writer.flush()
    writer.close()

    print("Generating visualization...")
    test_true_denorm = (test_true + 1) / 2 * (target_max - target_min) + target_min
    test_pred_denorm = (test_pred + 1) / 2 * (target_max - target_min) + target_min

    plt.figure(figsize=(15, 5))

    # Subplot 1: time-series comparison
    plt.subplot(1, 2, 1)
    plt.plot(test_true_denorm, label='Ground Truth', color='blue',   alpha=0.7)
    plt.plot(test_pred_denorm, label='Prediction',   color='orange', alpha=0.7)
    plt.title('Prediction vs Ground Truth')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(test_true_denorm, test_pred_denorm, alpha=0.5, s=1)
    plt.plot(
        [test_true_denorm.min(), test_true_denorm.max()],
        [test_true_denorm.min(), test_true_denorm.max()],
        'r--', lw=2
    )
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Scatter Plot (R2 = {test_r2:.3f})')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'out/{args.project_name}/predicted_coexpress.png', dpi=300, bbox_inches='tight')
    plt.close() 
    print(f'Figure saved to: out/{args.project_name}/predicted_coexpress.png')

    print('Running inference on all edges...')
    model.eval()
    all_predictions = []
    all_edge_correlations = []
    all_has_labels = []

    inference_start_time = time.time()

    with torch.no_grad():
        num_edges = data.edge_index.shape[1]

        for i in tqdm(range(0, num_edges, args.batch_size), desc='Inferring all edges'):
            end_idx = min(i + args.batch_size, num_edges)
            batch_edges = data.edge_index[:, i:end_idx]
            batch_edge_attr = data.edge_attr[i:end_idx]
            batch_has_labels = data.label_mask[i:end_idx]

            unique_nodes = torch.unique(batch_edges.flatten())

            try:
                loader = trainer.create_neighbor_loader(data, input_nodes=unique_nodes, shuffle=False)
                batch_data = next(iter(loader)).to(device)

                node_embeddings = model(batch_data.x, batch_data.edge_index)
                node_mapping = {node.item(): idx for idx, node in enumerate(batch_data.n_id)}

                valid_edges = []
                valid_edge_attr = []
                valid_has_labels = []

                for j in range(batch_edges.shape[1]):
                    src = batch_edges[0, j].item()
                    dst = batch_edges[1, j].item()
                    if src in node_mapping and dst in node_mapping:
                        valid_edges.append([node_mapping[src], node_mapping[dst]])
                        valid_edge_attr.append(batch_edge_attr[j])
                        valid_has_labels.append(batch_has_labels[j])

                if len(valid_edges) > 0:
                    mapped = torch.tensor(valid_edges, dtype=torch.long).t().contiguous().to(device)
                    ea_t = torch.stack(valid_edge_attr).to(device)
                    pred_w = model.predict_edges(node_embeddings, mapped, ea_t)
                    all_predictions.extend(pred_w.cpu().numpy())
                    all_edge_correlations.extend(ea_t.cpu().numpy().flatten())
                    all_has_labels.extend([x.item() for x in valid_has_labels])
                else:
                    n = end_idx - i
                    all_predictions.extend([0.0] * n)
                    all_edge_correlations.extend([0.0] * n)
                    all_has_labels.extend([False] * n)

                missing = (end_idx - i) - len(valid_edges)
                if missing > 0:
                    all_predictions.extend([0.0] * missing)
                    all_edge_correlations.extend([0.0] * missing)
                    all_has_labels.extend([False] * missing)

            except Exception as e:
                print(f'Error in batch {i // args.batch_size}: {e}')
                n = end_idx - i
                all_predictions.extend([0.0] * n)
                all_edge_correlations.extend([0.0] * n)
                all_has_labels.extend([False] * n)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    inference_end_time = time.time()
    inference_duration = inference_end_time - inference_start_time

    all_predictions = np.array(all_predictions)
    all_edge_correlations = np.array(all_edge_correlations)
    all_has_labels = np.array(all_has_labels)

    print(f'Inference complete (elapsed: {inference_duration:.2f} s):')
    print(f'Total edges: {len(all_predictions)}')
    print(f'Labeled edges: {all_has_labels.sum()}')
    print(f'Unlabeled edges:{(~all_has_labels).sum()}')
    print(f'Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]')

    print("Saving intermediate results...")
    pred_denormalized = (all_predictions + 1) / 2 * (target_max - target_min) + target_min
    sample_size = min(20000, len(all_predictions))
    sample_indices = np.random.choice(len(all_predictions), sample_size, replace=False)

    results = {
        'model_state_dict': model.state_dict(),
        'gene_to_idx': gene_to_idx,
        'all_nodes_genes': all_nodes_genes[:1000] if len(all_nodes_genes) > 1000 else all_nodes_genes,
        'common_genes': common_genes[:1000]    if len(common_genes)    > 1000 else common_genes,
        'normalization_params': {
            'target_min': target_min,
            'target_max': target_max,
            'edge_corr_min': edge_corr_min,
            'edge_corr_max': edge_corr_max,
        },
        'test_metrics': {'mse': test_mse, 'mae': test_mae, 'r2': test_r2},
        'training_stats': {
            'total_nodes': len(all_nodes_genes),
            'common_nodes': len(common_genes),
            'unlabeled_nodes': len(all_nodes_genes) - len(common_genes),
            'total_edges': len(all_predictions),
            'labeled_edges': int(all_has_labels.sum()),
            'unlabeled_edges': int((~all_has_labels).sum()),
        },
        'config': {
            'training': not args.only_eval,
            'top_k_edges': args.top_k_edges,
            'hidden_dims': args.hidden_dims,
            'edge_dim': args.edge_dim,
            'batch_size': args.batch_size,
            'num_neighbors': args.num_neighbors,
            'use_edge_features': True,
        },
        'timing': {
            'training_duration': training_duration,
            'inference_duration': inference_duration,
        },
        'sample_predictions': {
            'predictions':    all_predictions[sample_indices],
            'predictions_denormalized': pred_denormalized[sample_indices],
            'edge_correlations': all_edge_correlations[sample_indices],
            'has_labels':     all_has_labels[sample_indices],
            'indices':        sample_indices,
        },
    }

    with open(f'out/{args.project_name}/predicted_coexpress.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f'Intermediate results saved to: out/{args.project_name}/predicted_coexpress.pkl')

    print("Generating full co-expression matrix...")
    print('=' * 60)
    print('Saving results')
    print('=' * 60)

    save_start_time = time.time()

    result_df, save_stats = infer_coexpression(data, all_nodes_genes, pred_denormalized, args.coexpression_file, gene_to_idx, save_path=f'out/{args.project_name}/predicted_coexpress.pkl')
    save_end_time = time.time()
    save_duration = save_end_time - save_start_time

    results['save_statistics'] = save_stats
    results['timing']['save_duration'] = save_duration

    with open(f'out/{args.project_name}/predicted_coexpress.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f'Full results saved to: out/{args.project_name}/predicted_coexpress.pkl')

    print("Analysis Report:")
    print_analysis(save_stats)

    total_genes = save_stats['total_genes']
    orig_edges = save_stats['original_edges']
    pred_new = save_stats['labeled_pred_edges'] + save_stats['unlabeled_pred_edges']

    print('Training complete!')
    print('Key results:')
    print(f'1. Built a full co-expression network with {total_genes:,} genes')
    print(f'2. Retained all {orig_edges:,} original high-quality edges')
    print(f'3. Added {pred_new:,} predicted edges')
    print(f'4. Achieved R\u00b2 = {test_r2:.4f}')


def build_argparser():
    parser = argparse.ArgumentParser(description="Train Coxformer for gene co-expression prediction.")
    
    parser.add_argument('--project_name', type=str, default='top50_full_1e-3_L2', help='Project name for output folders')
    parser.add_argument('--top_k_edges', type=int, default=50, help='Number of top-k edges to keep per node')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[20, 15], help='Number of neighbors for sampling')
    parser.add_argument('--only_eval', action='store_true', help='Skip training and only run evaluation/inference')
    
    parser.add_argument('--coexpression_file', type=str, default='data/coexpression.pkl', help='Path to coexpression data')
    parser.add_argument('--correlation_file', type=str, default='data/sc_correlation_origin.pkl', help='Path to correlation data')
    parser.add_argument('--embedding_file', type=str, default='data/GPT_embedding.pkl', help='Path to embedding data')
    
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64], help='Hidden layer dimensions')
    parser.add_argument('--edge_dim', type=int, default=1, help='Edge feature dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    
    args = parser.parse_args()

    print(f'Project name: {args.project_name}')
    print(f'Top-k edges: {args.top_k_edges}')
    print(f'Neighbor samples: {args.num_neighbors}')
    print(f'Batch size: {args.batch_size}')
    print(f'Only eval: {args.only_eval}')

    return args


if __name__ == '__main__':
    args = build_argparser()
    main(args)