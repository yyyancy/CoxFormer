import torch
import numpy as np
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


class CoxformerTrainer:
    """
    Enhanced sampling trainer with support for partially labeled edges
    """
    def __init__(self, model, device, num_neighbors=[15, 10], batch_size=256):
        self.model = model.to(device)
        self.device = device
        self.num_neighbors = num_neighbors
        self.batch_size = batch_size
    
    def create_neighbor_loader(self, data, input_nodes=None, shuffle=True):
        """
        Create NeighborLoader for sampling
        """
        if input_nodes is None:
            input_nodes = torch.arange(data.x.shape[0])
        
        loader = NeighborLoader(
            data,
            num_neighbors=self.num_neighbors,
            input_nodes=input_nodes,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        
        return loader
    
    def train_epoch(self, data, optimizer, criterion, train_edge_indices):
        """
        Enhanced training: all edges participate in forward propagation, only labeled edges compute loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        num_labeled_edges = 0
        
        # Random shuffle training edges
        np.random.shuffle(train_edge_indices)
        
        # Process training edges in batches
        for i in range(0, len(train_edge_indices), self.batch_size):
            end_idx = min(i + self.batch_size, len(train_edge_indices))
            batch_edge_indices = train_edge_indices[i:end_idx]
            
            # Get batch data
            batch_edges = data.edge_index[:, batch_edge_indices]
            batch_edge_attr = data.edge_attr[batch_edge_indices].to(self.device)
            batch_targets = data.y[batch_edge_indices].to(self.device)
            batch_label_mask = data.label_mask[batch_edge_indices].to(self.device)
            
            # Only labeled edges compute loss
            if not batch_label_mask.any():
                continue  # Skip if this batch has no labeled edges
            
            # Get all nodes involved in edges
            unique_nodes = torch.unique(batch_edges.flatten())
            
            try:
                loader = self.create_neighbor_loader(data, input_nodes=unique_nodes, shuffle=False)
                batch_data = next(iter(loader))
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward propagation (all edges participate)
                node_embeddings = self.model(batch_data.x, batch_data.edge_index)
                
                # Node index mapping
                node_mapping = {node.item(): i for i, node in enumerate(batch_data.n_id)}
                
                # Process all edges
                valid_edges = []
                valid_edge_attr = []
                valid_targets = []
                valid_label_mask = []
                
                for j in range(batch_edges.shape[1]):
                    src_node = batch_edges[0, j].item()
                    dst_node = batch_edges[1, j].item()
                    
                    if src_node in node_mapping and dst_node in node_mapping:
                        valid_edges.append([node_mapping[src_node], node_mapping[dst_node]])
                        valid_edge_attr.append(batch_edge_attr[j])
                        valid_targets.append(batch_targets[j])
                        valid_label_mask.append(batch_label_mask[j])
                
                if len(valid_edges) == 0:
                    continue
                
                # Convert to tensors
                batch_edges_mapped = torch.tensor(valid_edges, dtype=torch.long).t().contiguous().to(self.device)
                valid_edge_attr_tensor = torch.stack(valid_edge_attr).to(self.device)
                valid_targets_tensor = torch.stack(valid_targets).to(self.device)
                valid_label_mask_tensor = torch.stack(valid_label_mask).to(self.device)
                
                # Predict edge weights for all edges
                pred_weights = self.model.predict_edges(node_embeddings, batch_edges_mapped, valid_edge_attr_tensor)
                
                # Compute loss only for labeled edges
                labeled_indices = valid_label_mask_tensor.nonzero().squeeze()
                if labeled_indices.numel() == 0:
                    continue
                
                if labeled_indices.dim() == 0:
                    labeled_indices = labeled_indices.unsqueeze(0)
                
                labeled_pred = pred_weights[labeled_indices]
                labeled_targets = valid_targets_tensor[labeled_indices]
                
                # Compute loss
                loss = criterion(labeled_pred, labeled_targets)
                
                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_labeled_edges += len(labeled_indices)
                num_batches += 1
                
            except Exception as e:
                print(f"Training batch {i//self.batch_size} error: {e}")
                continue
            
            # Clean GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"This epoch: {num_batches} batches, {num_labeled_edges} labeled edges")
        return avg_loss
    
    def evaluate(self, data, criterion, eval_edge_indices):
        """
        Evaluation: only evaluate labeled edges
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_edge_indices), self.batch_size), desc="Evaluating"):
                end_idx = min(i + self.batch_size, len(eval_edge_indices))
                batch_edge_indices = eval_edge_indices[i:end_idx]
                
                # Get batch data
                batch_edges = data.edge_index[:, batch_edge_indices]
                batch_edge_attr = data.edge_attr[batch_edge_indices]
                batch_targets = data.y[batch_edge_indices]
                batch_label_mask = data.label_mask[batch_edge_indices]
                
                # Only process labeled edges
                if not batch_label_mask.any():
                    continue
                
                unique_nodes = torch.unique(batch_edges.flatten())
                
                try:
                    loader = self.create_neighbor_loader(data, input_nodes=unique_nodes, shuffle=False)
                    batch_data = next(iter(loader))
                    batch_data = batch_data.to(self.device)
                    
                    node_embeddings = self.model(batch_data.x, batch_data.edge_index)
                    node_mapping = {node.item(): i for i, node in enumerate(batch_data.n_id)}
                    
                    valid_edges = []
                    valid_edge_attr = []
                    valid_targets = []
                    valid_label_mask = []
                    
                    for j in range(batch_edges.shape[1]):
                        src_node = batch_edges[0, j].item()
                        dst_node = batch_edges[1, j].item()
                        
                        if src_node in node_mapping and dst_node in node_mapping:
                            valid_edges.append([node_mapping[src_node], node_mapping[dst_node]])
                            valid_edge_attr.append(batch_edge_attr[j])
                            valid_targets.append(batch_targets[j])
                            valid_label_mask.append(batch_label_mask[j])
                    
                    if len(valid_edges) == 0:
                        continue
                    
                    batch_edges_mapped = torch.tensor(valid_edges, dtype=torch.long).t().contiguous().to(self.device)
                    valid_edge_attr_tensor = torch.stack(valid_edge_attr).to(self.device)
                    valid_targets_tensor = torch.stack(valid_targets).to(self.device)
                    valid_label_mask_tensor = torch.stack(valid_label_mask).to(self.device)
                    
                    pred_weights = self.model.predict_edges(node_embeddings, batch_edges_mapped, valid_edge_attr_tensor)
                    
                    # Only evaluate labeled edges
                    labeled_indices = valid_label_mask_tensor.nonzero().squeeze()
                    if labeled_indices.numel() == 0:
                        continue
                    
                    if labeled_indices.dim() == 0:
                        labeled_indices = labeled_indices.unsqueeze(0)
                    
                    labeled_pred = pred_weights[labeled_indices]
                    labeled_targets = valid_targets_tensor[labeled_indices]
                    
                    loss = criterion(labeled_pred, labeled_targets)
                    
                    total_loss += loss.item()
                    all_preds.extend(labeled_pred.cpu().numpy())
                    all_targets.extend(labeled_targets.cpu().numpy())
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Evaluation batch {i//self.batch_size} error: {e}")
                    continue
        
        # Compute evaluation metrics
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            mse = mean_squared_error(all_targets, all_preds)
            mae = mean_absolute_error(all_targets, all_preds)
            r2 = r2_score(all_targets, all_preds)
            
            return avg_loss, mse, mae, r2, all_preds, all_targets
        else:
            return float('inf'), float('inf'), float('inf'), -float('inf'), [], []
