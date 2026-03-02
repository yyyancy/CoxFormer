import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score, 
    recall_score
)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def preprocess_adata(adata, min_counts=20, n_top_genes=1000):
    if np.max(adata.X) > 100:
        print("Preprocess adata...")
        sc.pp.filter_genes(adata, min_counts=min_counts)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("Skip preprocess adata...")
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top_genes)
    adata = adata[:, adata.var['highly_variable']]
    return adata

def create_gene_embedding_matrix(adata, df_emb, embedding_column):
    gene_to_emb = {
        row['gene_name']: np.array(row[embedding_column])
        for _, row in df_emb.iterrows()
        if row['gene_name'] in adata.var_names
    }
    
    if len(gene_to_emb) == 0:
        raise ValueError("No overlapping genes found between AnnData and embeddings.")
    
    embedding_dim = len(next(iter(gene_to_emb.values())))
    gene_embedding_matrix = np.zeros((adata.n_vars, embedding_dim))
    count_missing = 0
    for i, gene in enumerate(adata.var_names):
        if gene in gene_to_emb:
            gene_embedding_matrix[i, :] = gene_to_emb[gene]
        else:
            count_missing+=1
    #print(f"embedding_dim:{embedding_dim}")
    #print(f"Unable to match {count_missing} in the embedding")

    # If embedding dimension > 2048, reduce to 512 via PCA
    if embedding_dim > 2048:
        pca = PCA(n_components=512, random_state=42)
        gene_embedding_matrix = pca.fit_transform(gene_embedding_matrix)
    return gene_embedding_matrix

def compute_cell_embeddings(adata, gene_embedding_matrix):
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    cell_embeddings = (X @ gene_embedding_matrix) / (X.sum(axis=1, keepdims=True) + 1e-12)
    return cell_embeddings

def load_and_preprocess_embeddings(method='genept', file_paths=None):
    df = pd.read_pickle(file_paths[method])
    print(f"\nColumns in '{method}' embeddings dataframe: {df.columns.tolist()}")

    # Detect embedding column
    possible_embedding_cols = ['embedding', 'Coexpress', 'GenePT', 'COPT','Copt','Embedding']
    embedding_col = None
    for col in possible_embedding_cols:
        if col in df.columns:
            embedding_col = col
            break
    if embedding_col is None:
        raise ValueError(f"No known embedding column found for method '{method}'")

    # Detect gene name column
    possible_gene_name_cols = ['gene_name', 'symbol', 'name', 'Gene name']
    gene_name_col = None
    for col in possible_gene_name_cols:
        if col in df.columns:
            gene_name_col = col
            break
    if gene_name_col is None:
        raise ValueError(f"No known gene name column found for method '{method}'")

    print(f"Using embedding column '{embedding_col}' and gene name column '{gene_name_col}' for method '{method}'.")

    # Filter out invalid embeddings
    df_filtered = df[df[embedding_col].apply(lambda x: np.sum(np.abs(np.array(x))) != 0)]
    df_filtered = df_filtered.reset_index(drop=True)

    if df_filtered.empty:
        raise ValueError(f"No valid embeddings found for method '{method}' after filtering.")

    print(df_filtered.head())
    return df_filtered, embedding_col, gene_name_col

def calculate_clustering_metrics(cell_embeddings, true_labels, pred_labels, cluster_method):
    unique_pred = np.unique(pred_labels)
    if len(unique_pred) < 2:
        # Not enough clusters to compute meaningful metrics
        ari, nmi, ami = np.nan, np.nan, np.nan
    else:
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ami = adjusted_mutual_info_score(true_labels, pred_labels)
    
    return {
        'Clustering_Method': cluster_method,
        'ARI': ari,
        'NMI': nmi,
        'AMI': ami
    }


def perform_clustering(cell_embeddings, true_labels):
    results = []
    fold_results = []
    # 使用 StratifiedKFold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ari = []
    nmi = []
    ami = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(cell_embeddings, true_labels)):
        X_train, X_test = cell_embeddings[train_idx], cell_embeddings[test_idx]
        y_train, y_test = true_labels[train_idx], true_labels[test_idx]

        # 对每个折进行 KMeans 聚类
        num_classes = len(np.unique(y_train))
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init='auto')
        pred_labels_kmeans_train = kmeans.fit_predict(X_train)

        # 对测试集进行预测
        pred_labels_kmeans_test = kmeans.predict(X_test)

        # 计算测试集的聚类指标
        unique_pred_test = np.unique(pred_labels_kmeans_test)
        if len(unique_pred_test) < 2:
            ari_test, nmi_test, ami_test= np.nan, np.nan, np.nan
        else:
            ari_test = adjusted_rand_score(y_test, pred_labels_kmeans_test)
            nmi_test = normalized_mutual_info_score(y_test, pred_labels_kmeans_test)
            ami_test = adjusted_mutual_info_score(y_test, pred_labels_kmeans_test)

        ari.append(ari_test)
        nmi.append(nmi_test)
        ami.append(ami_test)
        # 存储当前折的测试集聚类结果
        fold_results.append({
            'Fold': fold_idx + 1,
            'Clustering_Method': 'KMeans',
            'ARI': ari_test,
            'NMI': nmi_test,
            'AMI': ami_test,
        })

    # 返回最终的聚类指标结果（所有折的平均值）
    results.append({
        'Clustering': 'KMeans',
        'ARI': np.mean(ari),
        'NMI': np.mean(nmi),
        'AMI': np.mean(ami),
    })

    return results, fold_results

def perform_classification(cell_embeddings, true_labels):
    unique_labels = np.unique(true_labels)
    label_map = {u: i for i, u in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[l] for l in true_labels])
    multiclass = len(unique_labels) > 2
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fold_results = []
    classifiers = {
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=80),
    }

    for clf_name, clf in classifiers.items():
        auc = []
        accuracy = []
        f1 = []
        recall = []
        precision = []

        # For recording per-fold results
        fold_idx = 1
        for train_idx, test_idx in skf.split(cell_embeddings, numeric_labels):
            X_train, X_test = cell_embeddings[train_idx], cell_embeddings[test_idx]
            y_train, y_test = numeric_labels[train_idx], numeric_labels[test_idx]
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc_val = accuracy_score(y_test, y_pred)
            f1_val = f1_score(y_test, y_pred, average='macro')
            pre_val = precision_score(y_test, y_pred, average='macro')  # Precision
            recall_val = recall_score(y_test, y_pred, average='macro')  # Recall
   
            y_proba = clf.predict_proba(X_test)
            if multiclass:
                # Compute macro-average AUC for multiclass
                try:
                    y_test_bin = np.zeros((y_test.size, len(unique_labels)))
                    for i, label in enumerate(y_test):
                        y_test_bin[i, label] = 1
                
                    auc_per_class = []
                    for c in range(len(unique_labels)):
                        # Check if this class is present in y_test_bin
                        if np.sum(y_test_bin[:, c]) == 0 or np.sum(y_test_bin[:, c]) == y_test_bin.shape[0]:
                            # All zeros or all ones, not valid for ROC AUC
                            auc_c = np.nan
                        else:
                            auc_c = roc_auc_score(y_test_bin[:, c], y_proba[:, c])
                        auc_per_class.append(auc_c)
                    auc_val = np.nanmean(auc_per_class)
                except ValueError:
                    auc_val = np.nan
            else:
                # Binary classification AUC
                auc_val = roc_auc_score(y_test, y_proba[:, 1])

            auc.append(auc_val)
            accuracy.append(acc_val)
            f1.append(f1_val)
            precision.append(pre_val)
            recall.append(recall_val)

            # Store this fold's results
            fold_results.append({
                'Classifier': clf_name,
                'Fold': fold_idx,
                'AUC': auc_val,
                'Accuracy': acc_val,
                'F1': f1_val,
                'Precision':pre_val,
                'Recall':recall_val
            })
            fold_idx += 1

        # Aggregate results across folds
        results.append({
            'Classifier': clf_name,
            'AUC': np.mean(auc),
            'Accuracy': np.mean(accuracy),
            'F1': np.mean(f1),
            'Precision': np.mean(precision),
            'Recall': np.mean(recall)
        })

    return results, fold_results


def process_dataset_method(dataset_name, label_key, method, file_paths):
    # Load the dataset
    if dataset_name == 'bonemarrow':
        adata = scv.datasets.bonemarrow()
    elif dataset_name == 'BC4_FLEX':
        adata = sc.read_h5ad('Dataset/Cell_level_task/BC4_FLEX_converted.h5ad')
    elif dataset_name == 'DLBCL':
        adata = sc.read_h5ad('Dataset/Cell_level_task/DLBCL_converted.h5ad')
    elif dataset_name == 'Lung':
        adata = sc.read_h5ad('Dataset/Cell_level_task/Lung_converted.h5ad')
    else:
        adata = sc.read_h5ad(dataset_name)

    adata = preprocess_adata(adata, min_counts=20, n_top_genes=2000)
    
    true_labels = adata.obs[label_key].astype(str).values
    # Load embeddings
    df_emb, embedding_col, gene_name_col = load_and_preprocess_embeddings(method=method, file_paths=file_paths)

    # Create gene embedding matrix
    gene_embedding_matrix = create_gene_embedding_matrix(adata, df_emb, embedding_col)

    # Compute cell embeddings
    cell_embeddings = compute_cell_embeddings(adata, gene_embedding_matrix)

     # Classification metrics
    class_metrics_list, class_fold_metrics_list = perform_classification(cell_embeddings, true_labels)
    classification_rows = []
    for cml in class_metrics_list:
        row = {
            'Dataset': dataset_name,
            'Method': method
        }
        row.update(cml)
        classification_rows.append(row)

    # Store fold-wise classification results
    classification_fold_rows = []
    for cfl in class_fold_metrics_list:
        row = {
            'Dataset': dataset_name,
            'Method': method
        }
        row.update(cfl)
        classification_fold_rows.append(row)

    # Clustering metrics
    cluster_metrics_list, cluster_fold_metrics_list = perform_clustering(cell_embeddings, true_labels)
    clustering_rows = []
    for cm in cluster_metrics_list:
        row = {
            'Dataset': dataset_name,
            'Method': method
        }
        row.update(cm)
        clustering_rows.append(row)
    
    clustering_fold_rows = []
    for cf in cluster_fold_metrics_list:
        row = {
            'Dataset': dataset_name,
            'Method': method
        }
        row.update(cf)
        clustering_fold_rows.append(row)


    return clustering_rows, clustering_fold_rows, classification_rows, classification_fold_rows


def process_dataset_benchmark(dataset_name, label_key, method, file_paths):
    # Load the dataset
    if dataset_name == 'bonemarrow':
        adata = scv.datasets.bonemarrow()
    elif dataset_name == 'BC4_FLEX':
        adata = sc.read_h5ad('Dataset/Cell_level_task/BC4_FLEX_converted.h5ad')
    elif dataset_name == 'DLBCL':
        adata = sc.read_h5ad('Dataset/Cell_level_task/DLBCL_converted.h5ad')
    elif dataset_name == 'Lung':
        adata = sc.read_h5ad('Dataset/Cell_level_task/Lung_converted.h5ad')
    else:
        adata = sc.read_h5ad(dataset_name)
    
    adata = preprocess_adata(adata, min_counts=20, n_top_genes=1000)
    
    true_labels = adata.obs[label_key].astype(str).values
    # Load embeddings
    df_emb, embedding_col, gene_name_col = load_and_preprocess_embeddings(method=method, file_paths=file_paths)

    # Create gene embedding matrix
    gene_embedding_matrix = create_gene_embedding_matrix(adata, df_emb, embedding_col)

    # Compute cell embeddings
    cell_embeddings = compute_cell_embeddings(adata, gene_embedding_matrix)

    class_metrics_list, _ = perform_classification(cell_embeddings, true_labels)
    cluster_metrics_list, _ = perform_clustering(cell_embeddings, true_labels)

    merged_metrics = []
    assert len(class_metrics_list) == len(cluster_metrics_list), "len(class_metrics_list) != len(cluster_metrics_list)"
    for clf_result, cluster_result in zip(class_metrics_list, cluster_metrics_list):
        merged_result = clf_result.copy()
        merged_result.update(cluster_result)
        merged_metrics.append(merged_result)
    
    all_rows = []
    for cls in merged_metrics:
        row = {
            'Method': method
        }
        row.update(cls)
        all_rows.append(row)
    return all_rows


def process_cell_embedding(dataset_name, label_key, method, file_paths):
    # Load the dataset
    if dataset_name == 'bonemarrow':
        adata = scv.datasets.bonemarrow()
    elif dataset_name == 'BC4_FLEX':
        adata = sc.read_h5ad('Dataset/Cell_level_task/BC4_FLEX_converted.h5ad')
    elif dataset_name == 'DLBCL':
        adata = sc.read_h5ad('Dataset/Cell_level_task/DLBCL_converted.h5ad')
    elif dataset_name == 'Lung':
        adata = sc.read_h5ad('Dataset/Cell_level_task/Lung_converted.h5ad')
    else:
        adata = sc.read_h5ad(dataset_name)

    
    adata = preprocess_adata(adata, min_counts=20, n_top_genes=2000)
    
    true_labels = adata.obs[label_key].astype(str).values
    # Load embeddings
    df_emb, embedding_col, gene_name_col = load_and_preprocess_embeddings(method=method, file_paths=file_paths)

    # Create gene embedding matrix
    gene_embedding_matrix = create_gene_embedding_matrix(adata, df_emb, embedding_col)

    # Compute cell embeddings
    cell_embeddings = compute_cell_embeddings(adata, gene_embedding_matrix)

    cell_pkl = {}
    cell_pkl['embedding'] = cell_embeddings
    cell_pkl['label'] = true_labels
    if method == 'Coxexpression_cor_graph_top50_infer500_rd':
        method = 'copt'
    if method == 'genept_origin':
        method = 'genept'
    with open(f'cell_embedding/{dataset_name}_{method}.pkl', 'wb') as f:
        pickle.dump(cell_pkl, f)
        
    return cell_embeddings, true_labels
