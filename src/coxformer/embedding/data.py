import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import psutil
import os


class CoxformerDataset:
    """
    Enhanced large-scale graph data processing class, supporting training with all correlation edges
    """
    def __init__(self, coexpression_file, correlation_file, embedding_file):
        self.coexpression_file = coexpression_file
        self.correlation_file = correlation_file
        self.embedding_file = embedding_file
        
    def load_data(self):
        """
        Load data from files
        """
        print("Loading data...")
        
        with open(self.coexpression_file, 'rb') as f:
            coexp_data = pickle.load(f)
        
        with open(self.correlation_file, 'rb') as f:
            corr_data = pickle.load(f)
        
        with open(self.embedding_file, 'rb') as f:
            emb_data = pickle.load(f)
        
        print(f"Correlation data shape: {corr_data.shape}")
        print(f"Coexpression data shape: {coexp_data.shape}")
        print(f"Embedding data shape: {emb_data.shape}")
        
        return coexp_data, corr_data, emb_data
    
    def create_enhanced_graph_data(self, top_k_edges=30):
        """
        Create enhanced graph data:
        1. Use intersection of correlation and embedding as all nodes
        2. Use correlation data to build all edges
        3. Only edges with common genes (correlation+coexpression+embedding intersection) have labels
        4. Other edges participate in training but don't compute loss
        """
        coexp_data, corr_data, emb_data = self.load_data()
        
        # Find gene sets
        coexp_genes = set(coexp_data['gene_name'].values)
        corr_genes = set(corr_data['gene_name'].values)
        emb_genes = set(emb_data['gene_name'].values)
        
        # All nodes: intersection of correlation and embedding
        all_nodes_genes = list(corr_genes.intersection(emb_genes))
        print(f"All node genes count (corr ∩ emb): {len(all_nodes_genes)}")
        
        # Labeled genes: intersection of all three
        common_genes = list(coexp_genes.intersection(corr_genes).intersection(emb_genes))
        print(f"Labeled genes count (coexp ∩ corr ∩ emb): {len(common_genes)}")
        
        # Unlabeled but participating genes
        unlabeled_genes = list(set(all_nodes_genes) - set(common_genes))
        print(f"Unlabeled genes count: {len(unlabeled_genes)}")
        
        # Create gene to index mapping
        gene_to_idx = {gene: idx for idx, gene in enumerate(all_nodes_genes)}
        common_gene_indices = [gene_to_idx[gene] for gene in common_genes]
        
        # Prepare data
        print("Preparing node features...")
        emb_filtered = emb_data[emb_data['gene_name'].isin(all_nodes_genes)].copy()
        emb_filtered = emb_filtered.set_index('gene_name').loc[all_nodes_genes].reset_index()
        node_features = np.vstack(emb_filtered['Copt'].values)
        
        print("Preparing correlation data...")
        corr_filtered = corr_data[corr_data['gene_name'].isin(all_nodes_genes)].copy()
        corr_filtered = corr_filtered.set_index('gene_name').loc[all_nodes_genes].reset_index()
        
        print("Preparing coexpression data...")
        coexp_filtered = coexp_data[coexp_data['gene_name'].isin(common_genes)].copy()
        coexp_filtered = coexp_filtered.set_index('gene_name').loc[common_genes].reset_index()
        
        # Reorder correlation data columns
        print("Reordering correlation adjacency matrix...")
        corr_original_gene_order = corr_data['gene_name'].tolist()
        all_nodes_indices_in_corr = [corr_original_gene_order.index(gene) for gene in all_nodes_genes]
        
        for i in range(len(corr_filtered)):
            orig_corr_row = np.array(corr_filtered.iloc[i]['Coexpress'])
            reordered_corr = orig_corr_row[all_nodes_indices_in_corr]
            corr_filtered.at[i, 'Coexpress'] = reordered_corr
        
        # Reorder coexpression data columns
        print("Reordering coexpression adjacency matrix...")
        coexp_original_gene_order = coexp_data['gene_name'].tolist()
        common_indices_in_coexp = [coexp_original_gene_order.index(gene) for gene in common_genes]
        
        for i in range(len(coexp_filtered)):
            orig_coexp_row = np.array(coexp_filtered.iloc[i]['Coexpress'])
            reordered_coexp = orig_coexp_row[common_indices_in_coexp]
            coexp_filtered.at[i, 'Coexpress'] = reordered_coexp
        
        # Build graph edges
        print(f"Building graph edges, keeping top-{top_k_edges} edges per node...")
        
        edges = []
        edge_correlations = []
        edge_labels = []
        has_label_mask = []  # Mark which edges have labels
        
        # Create coexpression lookup dictionary
        coexp_dict = {}
        for i, row in enumerate(coexp_filtered.iterrows()):
            _, data_row = row
            gene_name = data_row['gene_name']
            gene_idx = gene_to_idx[gene_name]
            coexp_values = np.array(data_row['Coexpress'])
            
            for j, target_gene in enumerate(common_genes):
                if target_gene != gene_name and j < len(coexp_values):
                    target_idx = gene_to_idx[target_gene]
                    coexp_dict[(gene_idx, target_idx)] = coexp_values[j]
        
        print(f"Coexpression dictionary contains {len(coexp_dict)} edges")
        
        # Build edges for each node
        for i, row in tqdm(corr_filtered.iterrows(), total=len(corr_filtered), desc="Building edges"):
            gene_name = row['gene_name']
            gene_idx = gene_to_idx[gene_name]
            
            corr_values = np.array(row['Coexpress'])
            corr_values = corr_values[:len(all_nodes_genes)]
            
            # Exclude self-loops
            corr_values[gene_idx] = -np.inf
            
            # Select top-k edges
            abs_corr = np.abs(corr_values)
            top_k_indices = np.argsort(abs_corr)[-top_k_edges:]
            
            for j in top_k_indices:
                if j != gene_idx:
                    edges.append([gene_idx, j])
                    edge_correlations.append(corr_values[j])
                    
                    # Check if coexpression label exists
                    if (gene_idx, j) in coexp_dict:
                        edge_labels.append(coexp_dict[(gene_idx, j)])
                        has_label_mask.append(True)
                    else:
                        edge_labels.append(0.0)  # Placeholder
                        has_label_mask.append(False)
        
        print(f"Total edges: {len(edges)}")
        print(f"Labeled edges: {sum(has_label_mask)}")
        print(f"Unlabeled edges: {len(edges) - sum(has_label_mask)}")
        
        # Normalization
        edge_correlations = np.array(edge_correlations)
        edge_labels = np.array(edge_labels)
        has_label_mask = np.array(has_label_mask)
        
        # Normalize only labeled edges
        labeled_indices = np.where(has_label_mask)[0]
        if len(labeled_indices) > 0:
            labeled_targets = edge_labels[labeled_indices]
            target_min = labeled_targets.min()
            target_max = labeled_targets.max()
            edge_labels[labeled_indices] = 2 * (labeled_targets - target_min) / (target_max - target_min) - 1
        else:
            target_min = target_max = 0
        
        # Normalize edge features
        edge_corr_min = edge_correlations.min()
        edge_corr_max = edge_correlations.max()
        edge_correlations_norm = 2 * (edge_correlations - edge_corr_min) / (edge_corr_max - edge_corr_min) - 1
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        node_features = torch.tensor(node_features, dtype=torch.float32)
        edge_correlations_norm = torch.tensor(edge_correlations_norm, dtype=torch.float32)
        edge_labels = torch.tensor(edge_labels, dtype=torch.float32)
        has_label_mask = torch.tensor(has_label_mask, dtype=torch.bool)
        
        print(f"Graph statistics:")
        print(f"Number of nodes: {node_features.shape[0]}")
        print(f"Number of edges: {edge_index.shape[1]}")
        print(f"Labeled edges: {has_label_mask.sum().item()}")
        print(f"Average degree: {edge_index.shape[1] / node_features.shape[0]:.2f}")
        print(f"Edge feature range: [{edge_correlations_norm.min():.4f}, {edge_correlations_norm.max():.4f}]")
        print(f"Label range: [{edge_labels[has_label_mask].min():.4f}, {edge_labels[has_label_mask].max():.4f}]")
        
        # Create graph data
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_correlations_norm.unsqueeze(1),
            y=edge_labels,
            label_mask=has_label_mask  # Mark which edges have labels
        )
        
        return (data, gene_to_idx, all_nodes_genes, common_genes, 
                target_min, target_max, edge_corr_min, edge_corr_max)
    
    def print_memory_usage(self):
        """
        Print current memory usage
        """
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"Current memory usage: {memory_gb:.2f} GB")



def split_labeled_edges_indices(data):
    """
    Split labeled edge indices into train, validation, and test sets
    
    Args:
        data: PyG Data object
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
    
    Returns:
        train_indices, val_indices, test_indices: numpy arrays of edge indices
    """
    # Get indices of all labeled edges
    label_mask_np = data.label_mask.cpu().numpy()
    labeled_indices = np.where(label_mask_np)[0]
    
    print(f"Splitting labeled edges:")
    print(f"Total labeled edges: {len(labeled_indices)}")
    
    # Random shuffle
    np.random.shuffle(labeled_indices)

    train_indices = labeled_indices
    val_indices = train_indices
    test_indices = train_indices
    
    return train_indices, val_indices, test_indices