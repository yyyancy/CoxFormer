# ====== Basic ======
import os
import pickle
import gc

# ====== Numeric / Data ======
import numpy as np

# ====== Deep Learning ======
import torch
from torch.utils.data import DataLoader, TensorDataset

# ====== Progress ======
from tqdm import tqdm
from .utils import convert_to_dataframe, save_atomic


def predict_gene_expression(regressor, X_emb, test_loader, gene_name, condition, batch_size, device, save_dir):
    if not os.path.exists(save_dir + "_impute.csv"):
        X_emb = torch.tensor(X_emb, dtype=torch.float32).to(device)
        regressor.eval()
        all_predictions = []
        all_groundtruth = []
        with torch.no_grad():
            for batch in test_loader:
                batch_X, batch_y = batch
                pred = regressor(batch_X, condition) 
                all_predictions.append(pred.cpu().numpy())
                all_groundtruth.append(batch_y.cpu().numpy())
            
        # Concatenate all predictions
        y_pred = np.vstack(all_predictions)
        y_true = np.vstack(all_groundtruth)
    
        y_true_df, y_pred_df = convert_to_dataframe(y_true, y_pred, gene_name)
        y_pred_df.to_csv(save_dir + "_impute.csv", index=False)
    else:
        print(f"{save_dir + '_impute.csv'} exist, skip prediction...")
    return 

def predict_spot_expression(regressor, X_emb, test_loader, condition, batch_size, device, save_dir):
    if not os.path.exists(save_dir + "_pred.npy"):
        X_emb = torch.tensor(X_emb, dtype=torch.float32).to(device)
        regressor.eval()
        all_predictions = []
        all_groundtruth = []
        with torch.no_grad():
            for batch in test_loader:
                batch_x_idx, batch_img_idx, batch_y = batch
                bx = X_emb[batch_x_idx.squeeze(-1)]
                conditions = condition['image']
                bimg = conditions[batch_img_idx.squeeze(-1)]
                pred = regressor(bx, bimg)
                B = pred.shape[0]
                N = pred.shape[1] if pred.ndim >= 2 else 1
                pred = pred.view(B, N).sum(dim=1, keepdim=True)  # mean pooling
                batch_y = batch_y.view(-1, 1)    
                all_predictions.append(pred.cpu().numpy())
                all_groundtruth.append(batch_y.cpu().numpy())
            
        # Concatenate all predictions
        y_pred = np.vstack(all_predictions).reshape(-1)
        y_true = np.vstack(all_groundtruth).reshape(-1)
        np.save(save_dir + "_true.npy", y_true)
        np.save(save_dir + "_pred.npy", y_pred)
    else:
        print(f"{save_dir + '_pred.npy'} exist, skip prediction...")
    return

def predict_pixel_expression(regressor, X_embs, gene_to_idx, Histology_path, batch_size, device, save_dir):
    if not os.path.exists(save_dir + "_hyper.pkl"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        predictions_all = {}
        with open(Histology_path, 'rb') as file:
            image_embs = pickle.load(file)
        image_cls = np.stack(image_embs['cls'], axis=2)
        image_sub = np.stack(image_embs['sub'], axis=2)
        image_rgb = np.stack(image_embs['rgb'], axis=2)
        image = np.concatenate((image_cls, image_sub, image_rgb),axis=-1)
        px_num, py_num, n_features = image.shape
        image_flat = image.reshape(px_num*py_num,1,n_features)
        print(f"px_num:{px_num}, py_num:{py_num}, n_features:{n_features}")
        dataset = TensorDataset(torch.tensor(image_flat).to('cpu'))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i, (gene_name, idx) in enumerate(tqdm(gene_to_idx.items(), desc="Processing genes")): 
            sub_embs = X_embs[idx,:].reshape(1,-1)
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_image = batch[0].to(device)
                    batch_X = torch.tensor(np.repeat(sub_embs, batch_image.shape[0], axis=0)).to(device)
                    predicted = regressor(batch_X, batch_image)
                    predicted = predicted.cpu().numpy()
                    predictions.append(predicted)        
            y_pred = np.concatenate(predictions, axis=0).reshape(px_num,py_num)
            predictions_all[gene_name] = y_pred
        with open(save_dir + "_hyper.pkl", "wb") as f:
            pickle.dump(predictions_all, f)
    else:
        print(f"{save_dir + '_hyper.pkl'} exist, skip prediction...")
    return


def predict_cell_expression(regressor, X_embs, gene_to_idx, Histology_path, batch_size, device, save_dir):
    if not os.path.exists(save_dir + "_hyper_all.pkl.gz"):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        predictions_all = {}
        with open('/home/wangshu/project/yangyy/CoxFormer/Part2/0_Sum_Task/Dataset/HyperImpute/skin/cell2pixels.pkl', 'rb') as file:
            cell2pixels = pickle.load(file)
        cell_ids = list(cell2pixels.keys())
        with open(Histology_path, 'rb') as file:
            image_embs = pickle.load(file)
        image_cls = np.stack(image_embs['cls'], axis=2)
        image_sub = np.stack(image_embs['sub'], axis=2)
        image_rgb = np.stack(image_embs['rgb'], axis=2)
        image = np.concatenate((image_cls, image_sub, image_rgb),axis=-1)
        px_num, py_num, n_features = image.shape
        image_flat = image.reshape(px_num*py_num,1,n_features)
        print(f"px_num:{px_num}, py_num:{py_num}, n_features:{n_features}")
        dataset = TensorDataset(torch.tensor(image_flat).to('cpu'))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for gi, (gene_name, idx) in enumerate(tqdm(gene_to_idx.items(), desc="Processing genes")): 
            sub_embs = X_embs[idx,:].reshape(1,-1)
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch_image = batch[0].to(device)
                    batch_X = torch.tensor(np.repeat(sub_embs, batch_image.shape[0], axis=0)).to(device)
                    predicted = regressor(batch_X, batch_image)
                    predicted = predicted.cpu().numpy()
                    predictions.append(predicted)        
            y_pred = np.concatenate(predictions, axis=0).reshape(px_num,py_num)
            cell_vec = np.zeros(len(cell_ids))
            for i, cid in enumerate(cell_ids):
                pixels = cell2pixels[cid]
                if len(pixels) == 0:
                    continue
                rr, cc = zip(*pixels)
                rr = np.array(rr)
                cc = np.array(cc)
                cell_vec[i] = y_pred[rr, cc].sum()
            predictions_all[gene_name] = cell_vec.astype(np.float16)
            del predictions, y_pred, cell_vec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        save_atomic(predictions_all, save_dir + "_hyper_all.pkl.gz")
    else:
        print(f"{save_dir + '_hyper_all.pkl.gz'} exist, skip prediction...")
    return