import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Import SVM classifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_and_preprocess_embeddings(method='genept', file_paths=None):
    """
    Load and preprocess different types of gene embeddings.

    Returns:
    - df_filtered (pd.DataFrame): DataFrame with filtered embeddings.
    - embedding_col (str): Name of the embedding column.
    - gene_name_col (str): Name of the gene name column.
    """
    if method not in file_paths:
        raise ValueError(f"Method '{method}' not found in file_paths.")

    # Load data
    df = pd.read_pickle(file_paths[method])
    
    # Print columns for debugging
    print(f"\nColumns in '{method}' embeddings dataframe: {df.columns.tolist()}")

    # Detect embedding column
    possible_embedding_cols = ['embedding', 'Coexpress', 'GenePT', 'COPT', 'Copt', 'Embedding']
    embedding_col = None
    for col in possible_embedding_cols:
        if col in df.columns:
            embedding_col = col
            break
    if embedding_col is None:
        raise ValueError(f"No known embedding column found in dataframe for method '{method}'")

    # Detect gene name column
    possible_gene_name_cols = ['gene_name', 'symbol', 'name', 'Gene name']
    gene_name_col = None
    for col in possible_gene_name_cols:
        if col in df.columns:
            gene_name_col = col
            break
    if gene_name_col is None:
        raise ValueError(f"No known gene name column found in dataframe for method '{method}'")

    print(f"Using embedding column '{embedding_col}' and gene name column '{gene_name_col}' for method '{method}'.")

    # Preprocess embeddings
    df_filtered = df[df[embedding_col].apply(lambda x: np.sum(np.abs(np.array(x))) != 0)]
    df_filtered = df_filtered.reset_index(drop=True)

    if df_filtered.empty:
        raise ValueError(f"No valid embeddings found for method '{method}' after filtering.")

    # Print first few rows for debugging
    print(df_filtered.head())

    return df_filtered, embedding_col, gene_name_col

def evaluate_embedding_method(method, data_dict, positive_label, negative_label, file_paths=None):
    """
    Evaluate a single embedding method using Random Forest, Logistic Regression, and SVM.
    """
    # Load and preprocess embeddings
    df_filtered, embedding_col, gene_name_col = load_and_preprocess_embeddings(method, file_paths=file_paths)
    gene_to_embedding = dict(zip(df_filtered[gene_name_col], df_filtered[embedding_col]))

    # Print keys in gene_to_embedding for debugging
    print(f"First 5 keys in gene_to_embedding for method '{method}': {list(gene_to_embedding.keys())[:5]}")

    # Extract genes
    pos_genes = data_dict[positive_label]
    neg_genes = data_dict[negative_label]

    # Prepare data
    X = []
    y = []
    missing_genes = []
    for gene in pos_genes:
        if gene in gene_to_embedding:
            X.append(gene_to_embedding[gene])
            y.append(1)
        else:
            missing_genes.append(gene)
    for gene in neg_genes:
        if gene in gene_to_embedding:
            X.append(gene_to_embedding[gene])
            y.append(0)
        else:
            missing_genes.append(gene)

    if len(missing_genes) > 0:
        print(f"Warning: {len(missing_genes)} genes not found in embeddings for method '{method}'.")

    if len(X) == 0:
        raise ValueError(f"No matching genes found in embeddings for method '{method}'.")

    X = np.array(X)
    y = np.array(y)

    # Initialize models and cross-validation
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    lr_model = LogisticRegression(random_state=0, max_iter=1000)
    svm_model = SVC(kernel='rbf', probability=True, random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    rf_auc = []
    rf_acc = []
    rf_f1 = []
    rf_pre = []
    rf_rec = []
    
    lr_auc = []
    lr_acc = []
    lr_f1 = []
    lr_pre = []
    lr_rec = []

    svm_auc = []
    svm_acc = []
    svm_f1 = []
    svm_pre = []
    svm_rec = []

    # Perform cross-validation
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(list(X_train))
        X_val_scaled = scaler.transform(list(X_val))

        # Random Forest
        rf_model.fit(X_train_scaled, y_train)
        rf_proba = rf_model.predict_proba(X_val_scaled)[:, 1]
        rf_pred = (rf_proba >= 0.5).astype(int)   
        rf_auc.append(roc_auc_score(y_val, rf_proba))
        rf_acc.append(accuracy_score(y_val, rf_pred))
        rf_f1.append(f1_score(y_val, rf_pred))
        rf_pre.append(precision_score(y_val, rf_pred)) 
        rf_rec.append(recall_score(y_val, rf_pred))

        # Logistic Regression
        lr_model.fit(X_train_scaled, y_train)
        lr_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
        lr_pred = (lr_proba >= 0.5).astype(int)   
        lr_auc.append(roc_auc_score(y_val, lr_proba))
        lr_acc.append(accuracy_score(y_val, lr_pred))
        lr_f1.append(f1_score(y_val, lr_pred))
        lr_pre.append(precision_score(y_val, lr_pred)) 
        lr_rec.append(recall_score(y_val, lr_pred))
        
        # SVM
        svm_model.fit(X_train_scaled, y_train)
        svm_proba = svm_model.predict_proba(X_val_scaled)[:, 1]
        svm_pred = (svm_proba >= 0.5).astype(int)   
        svm_auc.append(roc_auc_score(y_val, svm_proba))
        svm_acc.append(accuracy_score(y_val, svm_pred))
        svm_f1.append(f1_score(y_val, svm_pred))
        svm_pre.append(precision_score(y_val, svm_pred)) 
        svm_rec.append(recall_score(y_val, svm_pred))

    return {
        'method': method,
        'rf_auc': rf_auc,
        'rf_acc': rf_acc,
        'rf_f1': rf_f1,
        'rf_pre': rf_pre,
        'rf_rec': rf_rec,
        'lr_auc': lr_auc,
        'lr_acc': lr_acc,
        'lr_f1': lr_f1,
        'lr_pre': lr_pre,
        'lr_rec': lr_rec,
        'svm_auc': svm_auc,
        'svm_acc': svm_acc,
        'svm_f1': svm_f1,
        'svm_pre': svm_pre,
        'svm_rec': svm_rec,
        'n_samples': len(y)
    }

def evaluate_embeddings(data_dict, positive_label, negative_label, methods, file_paths=None):
    """
    Evaluate all embedding methods and return results.
    """
    results = []

    for method in methods:
        print(f"\nEvaluating '{method}' embeddings...")
        try:
            result = evaluate_embedding_method(method, data_dict, positive_label, negative_label, file_paths)
            results.append(result)
        except ValueError as e:
            print(f"Error evaluating method '{method}': {e}")

    # Return results as DataFrame
    if results:
        return pd.DataFrame({
            'Method': [r['method'] for r in results],
            'RF_Mean_AUC': [np.mean(r['rf_auc']) for r in results],
            'RF_Std_AUC': [np.std(r['rf_auc']) for r in results],
            'RF_Mean_ACC': [np.mean(r['rf_acc']) for r in results],
            'RF_Std_ACC': [np.std(r['rf_acc']) for r in results],
            'RF_Mean_F1': [np.mean(r['rf_f1']) for r in results],
            'RF_Std_F1': [np.std(r['rf_f1']) for r in results],
            'RF_Mean_PRE': [np.mean(r['rf_pre']) for r in results],
            'RF_Std_PRE': [np.std(r['rf_pre']) for r in results],
            'RF_Mean_REC': [np.mean(r['rf_rec']) for r in results],
            'RF_Std_REC': [np.std(r['rf_rec']) for r in results],

            'LR_Mean_AUC': [np.mean(r['lr_auc']) for r in results],
            'LR_Std_AUC': [np.std(r['lr_auc']) for r in results],
            'LR_Mean_ACC': [np.mean(r['lr_acc']) for r in results],
            'LR_Std_ACC': [np.std(r['lr_acc']) for r in results],
            'LR_Mean_F1': [np.mean(r['lr_f1']) for r in results],
            'LR_Std_F1': [np.std(r['lr_f1']) for r in results],
            'LR_Mean_PRE': [np.mean(r['lr_pre']) for r in results],
            'LR_Std_PRE': [np.std(r['lr_pre']) for r in results],
            'LR_Mean_REC': [np.mean(r['lr_rec']) for r in results],
            'LR_Std_REC': [np.std(r['lr_rec']) for r in results],

            'SVM_Mean_AUC': [np.mean(r['svm_auc']) for r in results],
            'SVM_Std_AUC': [np.std(r['svm_auc']) for r in results],
            'SVM_Mean_ACC': [np.mean(r['svm_acc']) for r in results],
            'SVM_Std_ACC': [np.std(r['svm_acc']) for r in results],
            'SVM_Mean_F1': [np.mean(r['svm_f1']) for r in results],
            'SVM_Std_F1': [np.std(r['svm_f1']) for r in results],
            'SVM_Mean_PRE': [np.mean(r['svm_pre']) for r in results],
            'SVM_Std_PRE': [np.std(r['svm_pre']) for r in results],
            'SVM_Mean_REC': [np.mean(r['svm_rec']) for r in results],
            'SVM_Std_REC': [np.std(r['svm_rec']) for r in results],
        })
    else:
        return pd.DataFrame()