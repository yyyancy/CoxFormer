# ====== Basic ======
import os
import random
import pickle

# ====== Numeric / Data ======
import numpy as np
import pandas as pd
from scipy.sparse import issparse

# ====== ML ======
from sklearn.preprocessing import StandardScaler

# ====== Deep Learning ======
import torch
from torch.utils.data import  TensorDataset

# ====== Bio ======
import scanpy as sc
import anndata as ad


def read_spatial_data(Spatial_path):
    Spatial_data = pd.read_table(Spatial_path, sep='\t', header=0, index_col=0)
    Spatial_data.columns = Spatial_data.columns.astype(str).str.upper()
    Spatial_data.index = Spatial_data.index.astype(str).str.upper()
    return Spatial_data


def read_gene_embedding(Embedding_path):
    gene_embedding = pd.read_pickle(Embedding_path)
    print(f"Load {Embedding_path} successfully!")
    if 'Coexpress' in gene_embedding.columns:
        gene_embedding.rename(columns={'Coexpress': 'Embedding'}, inplace=True)
    elif 'GenePT' in gene_embedding.columns:
        gene_embedding.rename(columns={'GenePT': 'Embedding'}, inplace=True)
    gene_embedding = gene_embedding[gene_embedding['Embedding'].apply(lambda x: not np.sum(np.array(x)) == 0)].reset_index(drop=True)
    gene_embedding.index = gene_embedding['gene_name'].str.upper()
    return gene_embedding


def read_overlap_genes(Spatial_data,gene_embedding,dataset_idx,gene_path):
    if os.path.exists('Result/' + dataset_idx + '/overlap_genes.npy'):
        print('overlap_genes.npy exits, skip compute overlap_genes set.')  
        overlap_genes = np.load('Result/' + dataset_idx + '/overlap_genes.npy')
    else:
        with open(gene_path, 'r') as file:
            gene_list = [line.strip() for line in file.readlines()]
        gene_names = gene_embedding['gene_name'].values
        overlap_genes = list(set(gene_list) & set(gene_names) & set(Spatial_data.columns))
        if not os.path.exists('Result/' + dataset_idx):
            os.makedirs('Result/' + dataset_idx)
        np.save('Result/' + dataset_idx + '/overlap_genes.npy', overlap_genes)
    return overlap_genes


def read_condition(paths, gene_count, Pattern):
    if Pattern == 'none':
        return None
    else:
        condition = {}
        # add location
        location_path = paths['locs']
        if os.path.exists(location_path):
            try:
                locs_df = pd.read_table(location_path, header=0, index_col=0)
                location = locs_df[["x","y"]].values
            except Exception as e:
                print(f"[WARN] add spatial failed: {e}")
            condition['location'] = location
        
        # add image
        image_path = paths['hist']
        if os.path.exists(image_path):
            image = process_image_data(image_path,gene_count,Pattern)
            if 'x_idx' in image:
                condition['image_idx'] = image['x_idx']
            condition['image'] = image['x']           
        return condition
    

def process_embedding(gene_embedding):
    X_origin = np.array(gene_embedding["Embedding"].tolist(), dtype=float)
    X_scaler = StandardScaler()
    X_embs = X_scaler.fit_transform(X_origin)
    return X_embs

def process_index(paths, condition, indices_seen, indices_unseen, train_idx, Pattern):
    index_info = {}
    if Pattern == "spot":
        index_info['indices_seen'] = indices_seen
        index_info['indices_unseen'] = indices_unseen
    elif "pixel" in Pattern:
        slide_num = int(read_txt(paths["slide_num"])[0])
        index_info['spot_count'] = condition["location"].shape[0]
        index_info['train_idx'] = train_idx
        index_info['indices_seen'] = indices_seen
        index_info['indices_unseen'] = indices_unseen
        index_info['slide_num'] = slide_num
    return index_info


def process_image_data(image_path, seen_num, Pattern):
    with open(image_path, 'rb') as file:
        embs = pickle.load(file)
    emb_full = embs['x']
    if "pixel" not in Pattern:
        emb = np.mean(emb_full, axis=1)
        spots_num, seq_num, n_features = emb_full.shape 
        image_condition = {}
        idx = np.arange(emb.shape[0], dtype=np.int64).reshape(-1,1)
        image_condition["x"] = emb
    else:
        spots_num, seq_num, n_features = emb_full.shape             
        image_condition = {}
        idx = np.arange(emb_full.shape[0], dtype=np.int64).reshape(-1,1)
        image_condition["x"] = emb_full
        image_condition["x_idx"] = np.tile(idx, (seen_num, 1))
    return image_condition


def process_spatial_data(spatial_data, gene_embedding, dataset_idx, paths, Pattern, random_state, split_ratio):
    """
    Preprocess spatial data using the train/test split from AnnData
    """
    # Keep the original orientation where cells are rows and genes are columns
    X_mat = spatial_data.values 
    
    # Create AnnData object
    adata = ad.AnnData(
        X=X_mat,
        obs=pd.DataFrame(index=spatial_data.index), 
        var=pd.DataFrame(index=spatial_data.columns) 
    )
    # Add additional metadata
    adata.var_names_make_unique()  # Ensure unique gene names
    adata.obs_names_make_unique()  # Ensure unique cell names
    base_path = os.path.dirname(paths['locs'])
    
    if Pattern == "spot":
        X = adata.X
        dtype = X.dtype
    
        # ---- Case 1: integer dtype → definitely raw counts ----
        if np.issubdtype(dtype, np.integer):
            need_norm = True
            reason = f"integer dtype ({dtype})"
    
        # ---- Case 2: float dtype but nearly integer-valued ----
        elif np.issubdtype(dtype, np.floating):
            # sample subset to check if float values are nearly integers
            if issparse(X):
                sample_vals = X[:1000].toarray()
            else:
                sample_vals = X[:1000]
            frac_part = np.abs(sample_vals - np.round(sample_vals))
            int_like_ratio = (frac_part < 1e-6).sum() / sample_vals.size
            if int_like_ratio > 0.99:  # >99% values are integer-like
                need_norm = True
                reason = "float counts but nearly integer-valued"
            else:
                need_norm = False
                reason = "already normalized float values"
        else:
            need_norm = False
            reason = f"unknown dtype {dtype}"
    
        # ---- Apply normalization only if needed ----
        if need_norm:
            print(f"Data appears raw ({reason}), performing normalization...")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        else:
            print(f"Data already normalized ({reason}), skipping normalization.")
    
    elif "pixel" in Pattern:
        # Check if already 0-1 normalized
        cnts = adata.X
        cnts_min = np.nanmin(cnts)
        cnts_max = np.nanmax(cnts)
        if cnts_min >= 0 and cnts_max <= 1.1:
            print("Pixel data already scaled to [0, 1], skip normalization.")
        else:
            print(f"Normalizing pixel data (min {cnts_min:.3f}, max {cnts_max:.3f})...")
            cnts_min_axis = np.nanmin(cnts, axis=0)
            cnts_max_axis = np.nanmax(cnts, axis=0)
            cnts = (cnts - cnts_min_axis) / (cnts_max_axis - cnts_min_axis + 1e-12)
            adata.X = cnts
    
    else:
        raise ValueError(f"Unknown Pattern: {Pattern}")

    spatial_cols = set(adata.var_names)
    embed_idx = set(gene_embedding.index)
    train_genes, test_genes, overlap_genes = None, None, None

    if Pattern == "spot":
        # --- Case A: 若 genes_train 路径存在（只检查这一项）→ 采用文件划分 ---
        if os.path.exists(paths["genes_train"]):
            if not os.path.exists(paths["genes_test"]):
                raise ValueError("genes_train exist but genes_test not exist，please provide paths['genes_test'].")
    
            train_list = np.load(paths["genes_train"], allow_pickle=True)
            test_list  = np.load(paths["genes_test"], allow_pickle=True)
            gene_list  = list(set(train_list) | set(test_list)) 
    
            overlap_genes = list(set(gene_list) & spatial_cols & embed_idx)
            if len(overlap_genes) == 0:
                raise ValueError("train/test gene and Spatial/Embedding without overlap")
    
            train_genes = [g for g in train_list if g in overlap_genes]
            test_genes  = [g for g in test_list  if g in overlap_genes]
            adata.var["is_train"] = adata.var_names.isin(train_genes)
            adata.var["is_test"]  = adata.var_names.isin(test_genes)
            np.save(os.path.join(base_path, "genes_overlap.npy"), np.array(overlap_genes, dtype=object))
    
        # --- Case B: 无 train/test 文件，但有 genes_txt → 随机 8:2 人为划分并保存 ---
        elif os.path.exists(paths["genes_txt"]):
            gene_list = read_txt(paths["genes_txt"])
            overlap_genes = list(set(gene_list) & spatial_cols & embed_idx)
            if len(overlap_genes) == 0:
                raise ValueError("genes_txt and Spatial/Embedding without overlap.")
            rng = np.random.default_rng(random_state)
            og = overlap_genes.copy()
            rng.shuffle(og)
            split_idx = int(len(og) * split_ratio)
            train_genes = og[:split_idx]
            test_genes  = og[split_idx:]
            adata.var["is_train"] = adata.var_names.isin(train_genes)
            adata.var["is_test"]  = adata.var_names.isin(test_genes)
            np.save(os.path.join(base_path, "genes_overlap.npy"), np.array(overlap_genes, dtype=object))
            np.save(os.path.join(base_path, "genes_train.npy"), np.array(train_genes, dtype=object))
            np.save(os.path.join(base_path, "genes_test.npy"), np.array(test_genes, dtype=object))
    
        # --- Case C: 都没有 → 用 Spatial × Embedding 的交集；如无划分且 Pattern=spot 则再随机 8:2 ---
        else:
            overlap_genes = list(spatial_cols & embed_idx)
            if len(overlap_genes) == 0:
                raise ValueError("Spatial and Embedding without overlap.")
            # 若已有 is_train/is_test（从外部注入）就尊重；否则在 spot 场景下随机划分
            if Pattern == "spot" and (("is_train" not in adata.var.columns) or ("is_test" not in adata.var.columns)):
                rng = np.random.default_rng(random_state)
                og = overlap_genes.copy()
                rng.shuffle(og)
                split_idx = int(len(og) * split_ratio)
                train_genes = og[:split_idx]
                test_genes  = og[split_idx:]
                adata.var["is_train"] = adata.var_names.isin(train_genes)
                adata.var["is_test"]  = adata.var_names.isin(test_genes)
                gene_list  = list(set(train_genes) | set(test_genes))
                np.save(os.path.join(base_path, "genes_overlap.npy"), np.array(overlap_genes, dtype=object))
                np.save(os.path.join(base_path, "genes_train.npy"), np.array(train_genes, dtype=object))
                np.save(os.path.join(base_path, "genes_test.npy"), np.array(test_genes, dtype=object))
    
    elif "pixel" in Pattern:
        if os.path.exists(paths["genes_txt"]):
            gene_list = read_txt(paths["genes_txt"])
            overlap_genes = list(set(gene_list) & spatial_cols & embed_idx)
        else:
            overlap_genes = list(spatial_cols & embed_idx)
    else:
        raise ValueError(f"Unknown Pattern: {Pattern}")
    
    # Align data
    print(f"gene_embedding:{gene_embedding.shape}")
    gene_embedding_flt = gene_embedding.loc[[g for g in gene_embedding.index if g in overlap_genes]]
    
    #spatial_flt = spatial_data[gene_embedding_flt.index].reset_index(drop=True)
    gene_expression = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    gene_expression_flt = gene_expression[gene_embedding_flt.index]
    
    # Extract features and expression values
    X_origin = np.array(gene_embedding_flt["Embedding"].tolist(), dtype=float)
    X_scaler = StandardScaler()
    X_embs = X_scaler.fit_transform(X_origin)
    y = gene_expression_flt.values.T # Transpose to have genes as rows

    all_genes = list(gene_embedding_flt.index)
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    indices = [gene_to_idx[gene] for gene in all_genes]
    gene_to_idx_supple = {gene: idx for idx, gene in enumerate(list(gene_embedding.index))}
    print(f"gene_to_idx_supple:{len(gene_to_idx_supple)} (orig {len(gene_to_idx)})")

    if Pattern == "spot":
        if ("is_train" in adata.var.columns) and ("is_test" in adata.var.columns):
            train_genes = list(adata.var_names[adata.var["is_train"]].intersection(all_genes))
            test_genes  = list(adata.var_names[adata.var["is_test"] ].intersection(all_genes))
        else:
            train_genes = all_genes
            test_genes  = []
        indices_seen = [gene_to_idx[g] for g in train_genes]
        indices_unseen = [gene_to_idx[g] for g in test_genes]
        train_idx = None
        print(f"[INFO][spot] seen={len(indices_seen)} unseen={len(indices_unseen)}")
        return (X_embs, y, indices_seen, indices_unseen, train_idx, np.array(all_genes, dtype=object))
    
    elif "pixel" in Pattern:
        paths["genes_train"] = paths["genes_train"].replace("genes_train.npy", f"genes_{Pattern}_train.npy")
        paths["genes_test"] = paths["genes_test"].replace("genes_test.npy", f"genes_{Pattern}_test.npy")
        
        if Pattern == 'pixel_real':
            if os.path.exists(paths["genes_train"]):
                train_genes = np.load(paths["genes_train"], allow_pickle=True).tolist()
                test_genes  = np.load(paths["genes_test"], allow_pickle=True).tolist()
            else:
                nan_cols = np.where(np.isnan(y).any(axis=1))[0]
                test_genes = [gene for gene, idx in gene_to_idx.items() if idx in nan_cols]
                train_genes = [gene for gene, idx in gene_to_idx.items() if idx not in nan_cols]
                np.save(paths["genes_train"], np.array(train_genes, dtype=object))
                np.save(paths["genes_test"], np.array(test_genes, dtype=object))           
            gene_split = {
                    "train": {g: gene_to_idx[g] for g in train_genes if g in gene_to_idx},
                    "test":  {g: gene_to_idx[g] for g in test_genes  if g in gene_to_idx},
                    "pred":  {},
                    "all":  {g: gene_to_idx_supple[g] for g in gene_to_idx_supple if g not in train_genes}
                }
        
        elif Pattern == 'pixel_sim':
            if os.path.exists(paths["genes_train"]):
                print(f"None Nan Gene count: {len(np.where(~np.isnan(y).any(axis=1))[0])}")
                train_genes = np.load(paths["genes_train"], allow_pickle=True).tolist()
                test_genes  = np.load(paths["genes_test"],  allow_pickle=True).tolist()
                pred_genes  = np.load(os.path.join(base_path, f"genes_{Pattern}_pred.npy"),  allow_pickle=True).tolist()
                print(f"[pixel_sim] loaded files: train={len(train_genes)}, test={len(test_genes)}, pred={len(pred_genes)}")
            else:
                non_nan_idx = np.where(~np.isnan(y).any(axis=1))[0]
                non_nan_genes = [all_genes[i] for i in non_nan_idx]
        
                random.shuffle(non_nan_genes)
                n = len(non_nan_genes)
                print(f"n:{n}")
                s1 = int(0.6 * n)
                s2 = int(0.8 * n)
        
                train_genes = non_nan_genes[:s1]
                test_genes  = non_nan_genes[s1:s2]
                pred_genes  = non_nan_genes[s2:]
        
                print(f"[pixel_sim] split(non-NaN only): train={len(train_genes)}, test={len(test_genes)}, pred={len(pred_genes)}")
        
                np.save(paths["genes_train"], np.array(train_genes, dtype=object))
                np.save(paths["genes_test"],  np.array(test_genes,  dtype=object))
                np.save(os.path.join(base_path, f"genes_{Pattern}_pred.npy"),  np.array(pred_genes,  dtype=object))
        
            # 4) 统一输出结构（子映射）
            gene_split = {
                "train": {g: gene_to_idx[g] for g in train_genes if g in gene_to_idx},
                "test":  {g: gene_to_idx[g] for g in test_genes  if g in gene_to_idx},
                "pred":  {g: gene_to_idx[g] for g in pred_genes  if g in gene_to_idx},
            }

        genes_num, spots_num = y.shape  
        y_expanded = np.zeros((spots_num * genes_num, 1))  
        for gene_idx in range(genes_num):
            for spot_idx in range(spots_num):
                y_expanded[gene_idx * spots_num + spot_idx, :] = y[gene_idx, spot_idx]
        train_idx = np.repeat(np.array(indices).reshape(-1,1), spots_num, axis=0)
        return (X_embs, y_expanded, indices, gene_split, train_idx, np.array(all_genes))
    
    else:
        raise ValueError(f"Unknown Pattern: {Pattern}")
    

def normalize_spatial_coords(spatial_array):
    spatial_min = spatial_array.min(dim=0, keepdim=True).values
    spatial_max = spatial_array.max(dim=0, keepdim=True).values
    normed = (spatial_array - spatial_min) / (spatial_max - spatial_min)
    return normed * 2 - 1
    
    

def train_data_loader(X_embs, y, condition, index_info, all_genes, Pattern, Modality, save_dir, device):
    if Pattern == "spot":
        indices_seen = index_info['indices_seen']
        indices_unseen = index_info['indices_unseen']
        X_seen_tensor = torch.tensor(X_embs[indices_seen], dtype=torch.float32).to(device)
        y_seen_tensor = torch.tensor(y[indices_seen], dtype=torch.float32).to(device)
        X_unseen_tensor = torch.tensor(X_embs[indices_unseen], dtype=torch.float32).to(device)
        y_unseen_tensor = torch.tensor(y[indices_unseen], dtype=torch.float32).to(device)
        train_dataset = TensorDataset(
            torch.tensor(X_seen_tensor, dtype=torch.float32),
            torch.tensor(y_seen_tensor, dtype=torch.float32)
        )
        
        test_dataset = TensorDataset(
            torch.tensor(X_unseen_tensor, dtype=torch.float32),
            torch.tensor(y_unseen_tensor, dtype=torch.float32)
        )
        df_groundtruth = pd.DataFrame(y[indices_unseen].T, columns=[all_genes[i] for i in indices_unseen])
        gt_path = os.path.join(save_dir, "groundtruth.csv")
        df_groundtruth.to_csv(gt_path, index=False) 
        print(f"✅ Ground truth saved to: {gt_path}")

        print(f"X_train: {X_seen_tensor.shape},y_train: {y_seen_tensor.shape}")
        print(f"X_test: {X_unseen_tensor.shape},y_test: {y_unseen_tensor.shape}")
        condition_array = {}
        condition_dim = {}
        if Modality == 'location':
            spatial_array = torch.tensor(condition['location'], dtype=torch.float32).to(device)
            spatial_array = normalize_spatial_coords(spatial_array)
            condition_array['location'] = spatial_array.unsqueeze(0)
            condition_dim['location'] = spatial_array.shape[-1]
            condition_dim['image'] = 1
            condition_dim['none'] = y_seen_tensor.shape[-1]
            condition_array['none'] = None
            print(f"location_array: {condition_array['location'].shape}")
        
        elif Modality == 'image':
            image_array = torch.tensor(condition['image'], dtype=torch.float32).to(device)
            condition_array['image']  = image_array
            condition_dim['image'] = image_array.shape[-1]
            condition_dim['location'] = 1
            condition_dim['none'] = y_seen_tensor.shape[-1]
            condition_array['none'] = None
            print(f'image_array: {image_array.shape}')
    
        elif Modality == 'combine':
            spatial_array = torch.tensor(condition['location'], dtype=torch.float32).to(device)
            spatial_array = normalize_spatial_coords(spatial_array)
            image_array = torch.tensor(condition['image'], dtype=torch.float32).to(device)
            condition_array['location'] = spatial_array.unsqueeze(0)
            condition_array['image'] = image_array
            condition_dim['location'] = spatial_array.shape[-1]
            condition_dim['image'] = image_array.shape[-1]
            condition_dim['none'] = y_seen_tensor.shape[-1]
            condition_array['none'] = None
            print(f"location_array: {spatial_array.shape}")
            print(f'image_array: {image_array.shape}')

        else:
            condition_dim['none'] = y_seen_tensor.shape[-1]
            condition_array['none'] = None

    elif "pixel" in Pattern:
        # Convert input data to tensors
        indices_seen = index_info['indices_seen']
        X_seen_tensor = torch.tensor(X_embs[indices_seen], dtype=torch.float32).to(device)
        y_seen_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        # Get dimensions
        input_dim = X_seen_tensor.shape[-1]
        output_dim = y_seen_tensor.shape[-1]
        # Create dataset and dataloader
        condition_array = {}
        condition_dim = {}
        condition_array['image'] = torch.tensor(condition['image']).to(device)
        train_dataset, test_dataset, istar_data, condition_input_dim = split_pixel_data(X_seen_tensor, condition, y_seen_tensor, index_info, device)
            
        with open(save_dir + f"/istar_{Pattern}_data.pkl", "wb") as f:
            pickle.dump(istar_data, f)
        print(save_dir + f"/istar_{Pattern}_data.pkl saved succeffully!")
        condition_dim['image'] = condition_input_dim
        condition_dim['none'] = condition['image'].shape[-2]    
    return train_dataset, test_dataset, condition_dim, condition_array


def split_pixel_data(X, condition_seen, y, index_info, device):
    train_idx = torch.tensor(index_info['train_idx']).to(device)
    block_count = index_info['spot_count']
    slide_num = index_info['slide_num']
    gene_idx = index_info['indices_unseen']
    condition = torch.tensor(condition_seen['image_idx'], dtype=torch.int64).to('cpu')
    condition_dim = condition_seen['image'].shape[-1]
    # Initialize lists to hold the parts of the data
    train_idx_all, test_idx_all = [],[]
    train_condition_all, train_y_all = [], []
    test_condition_all, test_y_all = [], []
    extrain_Y_data_all, extest_Y_data_all  = [], []
    torch.manual_seed(42)
    random.seed(21)
    indices = torch.arange(block_count)
    train_indices = indices[slide_num:]
    test_indices = indices[:slide_num]

    for gene, idx in gene_idx['train'].items():
        train_idx_sub = train_idx[idx * block_count: (idx + 1) * block_count]
        gene_condition = condition[idx * block_count: (idx + 1) * block_count]
        gene_y = y[idx * block_count: (idx + 1) * block_count]
        train_idx_all.append(train_idx_sub)
        train_condition_all.append(gene_condition)
        train_y_all.append(gene_y)        

    for gene, idx in gene_idx['test'].items():
        idx_list = train_idx[idx * block_count: (idx + 1) * block_count]
        gene_condition = condition[idx * block_count: (idx + 1) * block_count]
        gene_y = y[idx * block_count: (idx + 1) * block_count]
        
        train_idx_all.append(idx_list[train_indices])
        train_condition_all.append(gene_condition[train_indices])
        train_y_all.append(gene_y[train_indices])
        extrain_Y_data_all.append(gene_y[train_indices])

        test_idx_all.append(idx_list[test_indices])
        test_condition_all.append(gene_condition[test_indices])
        test_y_all.append(gene_y[test_indices])
        extest_Y_data_all.append(gene_y[test_indices])

    for gene, idx in gene_idx['pred'].items():
        idx_list = train_idx[idx * block_count: (idx + 1) * block_count]
        gene_condition = condition[idx * block_count: (idx + 1) * block_count]
        gene_y = y[idx * block_count: (idx + 1) * block_count]
        
        test_idx_all.append(idx_list[test_indices])
        test_condition_all.append(gene_condition[test_indices])
        test_y_all.append(gene_y[test_indices])
        
    train_idx_combined = torch.cat(train_idx_all, dim=0)
    train_condition_combined = torch.cat(train_condition_all, dim=0)
    train_y_combined = torch.cat(train_y_all, dim=0)

    test_idx_combined = torch.cat(test_idx_all, dim=0)
    test_condition_combined = torch.cat(test_condition_all, dim=0)
    test_y_combined = torch.cat(test_y_all, dim=0)
   
    # Create TensorDatasets
    train_dataset = TensorDataset(train_idx_combined.to(device), train_condition_combined.to(device), train_y_combined.to(device))
    test_dataset = TensorDataset(test_idx_combined.to(device),test_condition_combined.to(device), test_y_combined.to(device))
    print(f"y_seen:{train_y_combined.shape},y_unseen:{test_y_combined.shape}")
    
    # Output the shapes of the datasets
    istar_data = {}
    istar_data['train_Y'] = extrain_Y_data_all
    istar_data['test_Y'] = extest_Y_data_all
    istar_data['seen_genes'] = list(gene_idx['test'].keys())
    return train_dataset, test_dataset, istar_data, condition_dim
