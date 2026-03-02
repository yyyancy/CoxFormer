import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from utils.Gene_activity_score_prediction_utils import get_topk_markers, order_spots_by_marker_score, plot_gaussian_heatmap
from matplotlib.gridspec import GridSpec
import gseapy as gp
import matplotlib.ticker as mtick
import anndata as ad
import re
import textwrap
import json
import ast

def find_max_resolution_for_two_clusters(adata, res_min=0.01, res_max=2.0, step=0.05, verbose=False):
    resolutions = np.arange(res_min, res_max + step, step)
    best_res = None
    prev_res = None
    prev_n = None
    for res in resolutions:
        key = f'leiden_{res:.2f}'
        sc.tl.leiden(adata, resolution=res, key_added=key)
        n_clusters = adata.obs[key].nunique()
        if verbose:
            print(f"resolution={res:.2f}, clusters={n_clusters}")
        if n_clusters == 2:
            best_res = res
        elif n_clusters > 2:
            if best_res is None:
                if verbose:
                    print(f"Jump detected: prev={prev_res}, now={res}, refining with step={step/10}")

                finer_min = prev_res if prev_res is not None else res_min
                finer_max = res
                finer_step = step / 10
                finer_resolutions = np.arange(finer_min, finer_max + finer_step, finer_step)

                for fres in finer_resolutions:
                    fkey = f'leiden_{fres:.4f}'
                    sc.tl.leiden(adata, resolution=fres, key_added=fkey)
                    fn = adata.obs[fkey].nunique()

                    if verbose:
                        print(f"finer search: res={fres:.4f}, clusters={fn}")

                    if fn == 2:
                        best_res = fres

                    elif fn > 2:
                        if verbose:
                            print(f"Stop searching at resolution={res:.2f} (clusters={n_clusters} > 2)")
                        break
                if verbose:
                    print("Finer search finished. Stop searching.")
                break

            if verbose:
                print(f"Stop searching at resolution={res:.2f} (>2 clusters)")
            break

        prev_res = res
        prev_n = n_clusters

    return best_res


def preprocess_and_cluster(adata_imp, n_neighbors=30, n_pcs=30):
    # 1. PCA (if not already performed)
    sc.tl.pca(adata_imp, svd_solver='arpack')
    
    # 2. Compute the neighborhood graph
    sc.pp.neighbors(adata_imp, n_neighbors=n_neighbors, n_pcs=n_pcs)
    adata_res = adata_imp.copy() 
    
    # 3. Leiden clustering
    resolution = find_max_resolution_for_two_clusters(adata_res, res_min=0.01, res_max=1, step=0.01, verbose=False)
    sc.tl.leiden(adata_imp, resolution=resolution)

    # Optional: UMAP visualization
    sc.tl.umap(adata_imp)
    sc.pl.umap(adata_imp, color='leiden')
    return adata_imp

def plot_spatial_adata(adata_imp, color=['leiden', 'ident.annot'], spot_size=320, norm_range=(-0.6, 0.1)):
    # Plot spatial data
    sc.pl.spatial(
        adata_imp,
        color=color, 
        img_key=None,
        spot_size=spot_size,
        show=False
    )

    # Get the current figure
    fig = plt.gcf()

    # Iterate over all subplots (Axes)
    for ax in fig.axes:
        # Set spine linewidth
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # Set tick font size for x and y axes
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        
        # Set label font size for x and y axes
        ax.xaxis.label.set_fontsize(15)
        ax.yaxis.label.set_fontsize(15)
        
        # Set title font size
        ax.title.set_fontsize(15)
        
        # Adjust legend font size
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(15)  # Legend text size
            legend.get_title().set_fontsize(15)  # Legend title size (if available)

    # Automatically adjust layout and display the plot
    plt.tight_layout()
    plt.show()

def evaluate_cluster_expression(adata_imp, pred_column='leiden', truth_column='ident.annot', tumor_keyword=['tumor']):
    
    X = adata_imp.X.A if hasattr(adata_imp.X, "A") else adata_imp.X
    df = pd.DataFrame(X, columns=adata_imp.var_names)
    df[pred_column] = adata_imp.obs[pred_column].values

    gene_means = df.groupby(pred_column).mean()
    abs_means = gene_means.abs().mean(axis=1)
    print(f"Mean absolute expression per cluster: {abs_means}")

    max_cluster = abs_means.idxmax()
    print(f"Cluster with the highest mean absolute expression: {max_cluster}")

    pred = (adata_imp.obs[pred_column] == max_cluster).astype(int).values

    assert truth_column in adata_imp.obs.columns, f"adata.obs lacks'{truth_column}'"
    pattern = '|'.join([kw.lower() for kw in tumor_keyword])
    truth = adata_imp.obs[truth_column].str.lower().str.contains(pattern).astype(int).values

    roc_auc = roc_auc_score(truth, pred)
    precision = precision_score(truth, pred)  
    recall = recall_score(truth, pred)      
    f1 = f1_score(truth, pred)            
    acc = accuracy_score(truth, pred)     

    fpr, tpr, _ = roc_curve(truth, pred)
    if roc_auc < 0.5:
        fpr, tpr = tpr, fpr 
        roc_auc = 1 - roc_auc 
  
    print(f"ROC_AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': acc,
        'fpr': fpr,
        'tpr': tpr,
    }

def run_cancer_abnormal_benchmark(
    datasets,
    save_dir,
    src_dir,
    methods=None,
):
    if methods is None:
        methods = ["CoxFormer", "gimVI", "SpaGE", "SpaOTsc", "Tangram",
                   "Seurat", "stPlus", "novoSpaRc", "LIGER"]

    for dataset_num in datasets:
        print(f"\n==================== {dataset_num} ====================")

        # ---- load basic files ----
        locs = pd.read_csv(os.path.join(src_dir, dataset_num, "locs.tsv"),
                           header=0, sep="\t", index_col=0)
        groundtruth = pd.read_csv(os.path.join(save_dir, dataset_num, "groundtruth.csv"), sep=",")
        spatial_adata = sc.read_h5ad(os.path.join(src_dir, dataset_num, f"{dataset_num}.h5ad"))

        genes_total  = np.load(os.path.join(src_dir, dataset_num, "genes_test.npy"), allow_pickle=True)
        genes_hk     = np.load(os.path.join(src_dir, dataset_num, "genes_val.npy"),  allow_pickle=True)
        genes_cancer = list(set(genes_total) - set(genes_hk))

        genes_order = [g for g in groundtruth.columns if g in genes_cancer]

        # ---- build AnnData for evaluation ----
        var_anno = spatial_adata.var.reindex(genes_order).copy()
        assert len(genes_order) == len(var_anno), "Gene annotation mismatch"

        adata_gt = ad.AnnData(
            X   = groundtruth[genes_order].values,
            obs = spatial_adata.obs.copy(),
            var = var_anno
        )
        assert locs.shape[0] == adata_gt.n_obs, "locs and obs size mismatch"
        adata_gt.obsm["spatial"] = locs[["x", "y"]].values

        # ---- run each method ----
        Evaluate = {}
        for Method in methods:
            print(f"--------------- {Method} ---------------")
            imp_path = os.path.join(save_dir, dataset_num, f"{Method}_impute.csv")
            if not os.path.exists(imp_path):
                print(f"[WARN] missing: {imp_path} -> skip")
                continue

            result = pd.read_csv(imp_path, sep=",")
            metrics = abnormal_detection(adata_gt, result, Method, genes_order, save_dir, dataset_num)
            Evaluate[Method] = metrics

        # ---- save metrics.csv ----
        if len(Evaluate) == 0:
            print(f"[WARN] no results for {dataset_num}")
            continue

        df = pd.DataFrame.from_dict(Evaluate, orient="index")
        df.index.name = "Method"

        # minimal safe conversion for array-like columns
        for col in ["fpr", "tpr"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(list(np.asarray(x).ravel())) if x is not None else ""
                )
        file_path = os.path.join(save_dir, dataset_num, "metrics.csv")
        if not os.path.exists(file_path):
            df.to_csv(file_path)
        else:
            print(f"{file_path} already exists, skip saving.")
    print("\nDone.")



def abnormal_detection(adata,result,Method,genes_order,save_dir,dataset_num):
    h5ad_path = os.path.join(save_dir,dataset_num, f"{dataset_num}_{Method}_deg.h5ad")
    if os.path.exists(h5ad_path):
        adata_imp = sc.read_h5ad(h5ad_path)
    else:
        # Add prediction layer
        genes_common = [g for g in genes_order if g in result.columns]
        ref = adata[:,genes_common].copy()
        ref.layers["predicted"] = result[genes_common].values
        
        target = ref.copy()
        target.X = target.layers["predicted"].copy()
        target.X[target.X < 0] = 0 
        
        # Add group labels
        ref.obs["group"] = "ref"
        target.obs["group"] = "target"
        
        # -------------------- DEG analysis --------------------
        deg_path = os.path.join(save_dir,dataset_num, f"{dataset_num}_{Method}_deg.npy")
        if os.path.exists(deg_path):
            print(f"{os.path.basename(deg_path)} loaded successfully.")
            deg = np.load(deg_path, allow_pickle=True)
        else:
            combined = sc.concat([target, ref], label="group", keys=["target", "ref"], merge="same")
            sc.tl.rank_genes_groups(combined, groupby="group", method="wilcoxon")
            sc.pl.rank_genes_groups(combined, n_genes=20, sharey=False)
            deg_df = sc.get.rank_genes_groups_df(combined, group="target")
            mask = (deg_df["pvals_adj"] < 1e-2) & (deg_df["logfoldchanges"].abs() > 3)
            deg = deg_df.loc[mask, "names"].values
            np.save(deg_path, deg)
            deg_df.to_csv(os.path.join(save_dir,dataset_num,f"{dataset_num}_{Method}_deg.csv"), index=True)
        
        # -------------------- Subset top DEGs --------------------
        top_k = 200
        deg_sub = [g for g in deg[:top_k] if g in adata.var_names]
        adata_imp = ref[:, deg_sub].copy()
        adata_imp.X = ref[:, deg_sub].X.copy() - target[:, deg_sub].X.copy()
        
        # Preprocess and visualize
        adata_imp = preprocess_and_cluster(adata_imp, n_neighbors=30, n_pcs=30)
        plot_spatial_adata(
            adata_imp, color=["leiden", "ident.annot"], spot_size=200, norm_range=(-0.6, 0.1)
        )
        adata_imp.write(os.path.join(save_dir,dataset_num, f"{dataset_num}_{Method}_deg.h5ad"))
    
    # -------------------- Evaluate --------------------
    metrics = evaluate_cluster_expression(
        adata_imp, pred_column="leiden", truth_column="ident.annot", tumor_keyword=["Tumor"]
    )
    return metrics

# ---------------- internal helpers ----------------
def to_float_array(x):
    return np.fromstring(str(x).strip("[]"), sep=" ")

def load_metrics_tables(base_dir, datasets):
    dfs = {}
    for ds in datasets:
        df = pd.read_csv(os.path.join(base_dir, ds, "metrics.csv"), index_col=0)
        dfs[ds] = df
    return dfs

def method_to_raw(method_show):
    """Map display name back to raw name in csv if needed."""
    inv = {v: k for k, v in _RENAME_MAP.items()}
    return inv.get(method_show, method_show)

# ---------------- 1) barplots ----------------
def plot_metrics_barplots(base_dir, datasets, color_map, save):
    """
    Barplots of mean±std across datasets for Accuracy/F1/Recall/Precision.
    Input:
      - base_dir: "Result/Cancer"
      - datasets: ["colon1", ...]
      - color_map: dict, keys are display method names (e.g., "CoxFormer", "SpaGE", ...)
    """
    if len(datasets) > 1:
        save_name = 'all'
    else:
        save_name = datasets[0]
    methods_show = list(color_map.keys())
    dfs = load_metrics_tables(base_dir, datasets)
    combined = pd.concat(dfs, axis=0)  # MultiIndex: (dataset, method_raw)

    metrics = ["precision", "recall", "f1_score", "accuracy", "roc_auc"]
    mean_df = combined[metrics].groupby(level=1).mean()
    std_df  = combined[metrics].groupby(level=1).std()

    mean_df = mean_df.loc[[m for m in methods_show if m in mean_df.index]]
    std_df  = std_df.loc[[m for m in methods_show if m in std_df.index]]

    normal_metrics = ["accuracy", "f1_score", "recall", "precision"]
    ylabel_map = {"precision": "Precision", "recall": "Recall", "f1_score": "F1", "accuracy": "Accuracy"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 6))
    axes = axes.flatten()

    for ax, metric in zip(axes, normal_metrics):
        order = mean_df[metric].sort_values(ascending=False).index
        vals = mean_df.loc[order, metric]
        errs = std_df.loc[order, metric]
        bar_colors = [color_map.get(m, "#999999") for m in order]

        ax.bar(order, vals, yerr=errs, capsize=3, color=bar_colors)
        ax.set_ylabel(ylabel_map[metric], fontsize=14)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    if save == True:
        plt.savefig(os.path.join('Result/Cancer', f'cancer_{save_name}_metric.pdf'), bbox_inches='tight', pad_inches=0.1) 
    plt.show()
    return mean_df

# ---------------- 2) mean ROC ----------------
def plot_mean_roc(base_dir, datasets, color_map, save):
    """
    Mean ROC curve across datasets (standalone).
    - methods are taken from color_map keys (display names)
    - rename_map is internal: {"CoxFormer-Loc":"CoxFormer"}
    - AUC is computed as the mean of per-dataset roc_auc values in metrics.csv
    """
    methods_show = list(color_map.keys())
    line_styles = [(0, (5, 1)),(0, (3, 1, 1, 1)),(0, (5, 10)),(0, (1, 1)),(0, (3, 5, 1, 5)),":","-.","--","-"]
    dfs = load_metrics_tables(base_dir, datasets)
    if len(datasets) > 1:
        save_name = 'all'
    else:
        save_name = datasets[0]

    fpr_base = np.linspace(0, 1, 10)
    avg_roc = {}
    mean_auc = {}

    for m_show in methods_show:
        all_tpr = []
        auc_list = []

        for ds in datasets:
            df = dfs[ds]
            fpr = df.loc[m_show, "fpr"]
            fpr = np.asarray(ast.literal_eval(fpr), dtype=float).ravel()       
            tpr = df.loc[m_show, "tpr"]
            tpr = np.asarray(ast.literal_eval(tpr), dtype=float).ravel()
            all_tpr.append(np.interp(fpr_base, fpr, tpr))
            if "roc_auc" in df.columns:
                auc_list.append(float(df.loc[m_show, "roc_auc"]))

        if len(all_tpr) > 0:
            avg_roc[m_show] = np.mean(np.vstack(all_tpr), axis=0)

        if len(auc_list) > 0:
            mean_auc[m_show] = float(np.mean(auc_list))
        else:
            mean_auc[m_show] = np.nan

    # sort by AUC (ascending, so best drawn last to avoid being covered)
    sorted_methods = sorted(avg_roc.keys(), key=lambda m: (np.nan_to_num(mean_auc.get(m, np.nan), nan=-1.0)))

    plt.figure(figsize=(4.5, 4))
    for i, m in enumerate(sorted_methods):
        auc = mean_auc.get(m, np.nan)
        label = f"{m} (AUC:{auc:.2f})" if np.isfinite(auc) else m
        plt.plot(
            fpr_base, avg_roc[m],
            color=color_map.get(m, "#999999"),
            linestyle=line_styles[i % len(line_styles)],
            lw=3,
            label=label
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, alpha=1)

    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    if save == True:
        plt.savefig(os.path.join('Result/Cancer', f'cancer_{save_name}_roc.pdf'),bbox_inches='tight')
    plt.show()
    return mean_auc

def plot_abnormal_region(save_dir, dataset_num, Methods,truth_column = "ident.annot",tumor_keyword = ["tumor"]):
    truth_column = "ident.annot"
    tumor_keyword = ["tumor"]
    pattern = "|".join([kw.lower() for kw in tumor_keyword])

    # --- load a reference AnnData for coordinates + truth ---
    adata_ref = sc.read_h5ad(os.path.join(save_dir, f"{dataset_num}_CoxFormer_deg.h5ad"))
    spatial = adata_ref.obsm["spatial"]
    x_coords, y_coords = spatial[:, 0], spatial[:, 1]

    truth = (
        adata_ref.obs[truth_column]
        .astype(str).str.lower()
        .str.contains(pattern)
        .astype(int)
        .values
    )
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6), dpi=300)

    cmap_gt = LinearSegmentedColormap.from_list("custom_cmap_gt", ["#9E9E9E", "#D87659"])
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#9E9E9E", "#F5C96B"])

    ax = axes[0,0]
    ax.scatter(x_coords, y_coords, c=truth, s=10, cmap=cmap_gt, alpha=1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')
        ax.set_title("Groundtruth", fontsize=18, fontweight="bold")

    for idx, Method in enumerate(Methods):

        row = (idx + 1) // 5    # 从第二格开始填
        col = (idx + 1) % 5
        ax = axes[row, col]

        adata_imp = sc.read_h5ad(os.path.join(save_dir, f"{dataset_num}_{Method}_deg.h5ad"))
        X = adata_imp.X.A if hasattr(adata_imp.X, "A") else adata_imp.X
        df = pd.DataFrame(X, columns=adata_imp.var_names)
        df['leiden'] = adata_imp.obs['leiden'].values

        gene_means = df.groupby('leiden').mean()
        abs_means = gene_means.abs().mean(axis=1)
        max_cluster = abs_means.idxmax()

        pred = (adata_imp.obs['leiden'] == max_cluster).astype(int).values

        pattern = '|'.join([kw.lower() for kw in tumor_keyword])
        truth = adata_imp.obs[truth_column].str.lower().str.contains(pattern).astype(int).values

        ax.scatter(x_coords, y_coords, c=pred, s=10, cmap=cmap, alpha=1)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(Method, fontsize=18, fontweight="bold")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

    plt.tight_layout()
    file_path = os.path.join(save_dir, f'{dataset_num}_cancer_heatmap.pdf')
    if not os.path.exists(file_path):
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    else:
        print(f"{file_path} already exists, skip saving.")
    plt.show()

def plot_umap(save_dir, dataset_num, Methods):
    assert len(Methods) == 9, f"期望 9 个方法，但现在是 {len(Methods)}：{Methods}"

    # 2行×5列：第1列(索引0)给大图；后面4列给8个小图(2×4)
    fig = plt.figure(figsize=(10, 3), dpi=300, constrained_layout=True)
    gs = GridSpec(
        nrows=2, ncols=5, figure=fig,
        width_ratios=[2, 1, 1, 1, 1],   # 大图更宽；可自己调
        height_ratios=[1, 1],
        wspace=0.1, hspace=0.1
    )

    ax_big = fig.add_subplot(gs[:, 0])  # 跨两行
    ax_smalls = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(1, 5)]  # 8个小图

    def _draw_one(ax, Method,s=2):
        adata_imp = sc.read_h5ad(os.path.join(save_dir, f"{dataset_num}_{Method}_deg.h5ad"))
        umap_data = adata_imp.obsm["X_umap"]
        leiden_labels = adata_imp.obs["leiden"].astype(str)

        X = adata_imp.X.A if hasattr(adata_imp.X, "A") else adata_imp.X
        df = pd.DataFrame(X, columns=adata_imp.var_names)
        df["leiden"] = adata_imp.obs["leiden"].values

        gene_means = df.groupby("leiden").mean()
        abs_means = gene_means.abs().mean(axis=1)
        max_cluster = str(abs_means.idxmax())

        unique_clusters = sorted(leiden_labels.unique())
        palette = {str(c): "#51B1B7" for c in unique_clusters}
        palette[max_cluster] = "#E07B54"

        sns.scatterplot(
            x=umap_data[:, 0],
            y=umap_data[:, 1],
            hue=leiden_labels,
            hue_order=unique_clusters,
            palette=palette,
            s=s,
            alpha=0.8,
            legend=False,
            edgecolor=None,
            ax=ax
        )
        ax.set_title(Method, fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    # 第一个方法画到大图，其余8个方法画到右侧8个小图
    _draw_one(ax_big, Methods[0],s=4)
    for ax, Method in zip(ax_smalls, Methods[1:]):
        _draw_one(ax, Method)

    file_path = os.path.join(save_dir, f"{dataset_num}_umap.pdf")
    if not os.path.exists(file_path):
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    else:
        print(f"{file_path} already exists, skip saving.")
    plt.show()


def modify_label(term,
                 lowercase_words={"of", "and", "in", "for", "to",
                                  "the", "on", "at", "by"},
                 max_chars=30,      # 每行最多多少字符
                 indent="    "):    # 第二行以后缩进用空格
    term = re.sub(r"\s*\(GO:\d+\)", "", term)
    words = term.split()
    last = len(words) - 1
    formatted = []
    for i, w in enumerate(words):
        wl = w.lower()
        if 0 < i < last and wl in lowercase_words:
            formatted.append(wl)
        elif len(wl) <= 3:
            formatted.append(w.upper())
        else:
            formatted.append(w.capitalize())
    term = " ".join(formatted)
    raw_lines = textwrap.wrap(term, width=max_chars,
                              break_long_words=True,
                              break_on_hyphens=False)
    out_lines = []
    for j, ln in enumerate(raw_lines):
        if j == 0:
            out_lines.append(ln)
        else:
            out_lines.append(indent + ln)
    wrapped = "-\n".join(out_lines)  
    return wrapped

def enrich_go(glist, lib="GO_Biological_Process_2021", org='human', topn=10):
    enr = gp.enrich(gene_list=list(set(glist)), gene_sets=os.path.join(os.getcwd(),"Dataset",f"{lib}.gmt"),
                    outdir=None,
                    cutoff=0.05)
    if enr is None or enr.results is None or enr.results.empty:
        return pd.DataFrame(columns=["Term","Adjusted P-value","Combined Score","Overlap"])
    res = enr.results[["Term","Adjusted P-value","Combined Score","Overlap"]].copy()
    res = res.sort_values(["Adjusted P-value","Combined Score"]).head(topn).reset_index(drop=True)
    return res

def GO_analysis(glist):
    go_enrich = enrich_go(glist)
    dfb = go_enrich.head(6).copy()
    if "Adjusted P-value" in dfb.columns:
        score = -np.log10(np.maximum(1e-300, dfb["Adjusted P-value"].values))
        score_label = "-log10 Adjusted P-value"
    elif "FDR" in dfb.columns:
        score = -np.log10(np.maximum(1e-300, dfb["FDR"].values))
        score_label = "-log10 FDR"
    else:
        score = -np.log10(np.maximum(1e-300, dfb.get("P", 1).values))
        score_label = "-log10 P"
    
    # 计算 Gene Count 和 Gene Ratio
    dfb["Gene Count"] = dfb["Overlap"].str.split("/").apply(lambda x: int(x[0]))  # Gene Count
    dfb["Gene Ratio"] = dfb["Overlap"].str.split("/").apply(lambda x: int(x[0]) / int(x[1]))  # Gene Ratio
    
    return dfb, score, score_label


def plot_go_bubble(ax, dfb, title, cmap):
    # ---- compute color value ----
    dfb = dfb.copy()
    dfb["-log10(P)"] = -np.log10(np.maximum(1e-300, dfb["Adjusted P-value"]))

    size_ratio = 30
    # ---- scatter (no clipping) ----
    sc = ax.scatter(
        dfb["Gene Ratio"], dfb["Term"],
        s=[count * size_ratio for count in dfb["Gene Count"]],
        c=dfb["-log10(P)"],
        cmap=cmap,
        edgecolor="k",
        alpha=1,
        clip_on=False, 
        zorder=3
    )
    
    cbar = ax.figure.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=12, length=6, width=2)
    cbar.outline.set_linewidth(2)
    # cbar.set_label("-log10(P.adjust)", fontsize=12, fontweight="bold")

    size_legend = [5, 10]
    for s in size_legend:
        ax.scatter([], [], s=s * size_ratio, color="gray", alpha=0.6, label=f"{s} genes")

    # ---- x ticks: keep 2 decimals but avoid duplicates ----
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    # ---- labels ----
    ax.set_xlabel("Gene Ratio", fontsize=12, fontweight="bold")
    #ax.set_title(title, fontsize=12, fontweight="bold")

    # ---- tick fontsize ----
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    new_labels = [modify_label(term, max_chars=25) for term in dfb["Term"]]
    ax.set_yticklabels(new_labels[::-1], fontsize=12, fontweight="bold")

    # ---- no grid ----
    ax.grid(False)

    # ---- margins / limits to prevent bubble cutoff ----
    ax.margins(y=0.15)
    x = dfb["Gene Ratio"].to_numpy(float)
    xmin, xmax = np.min(x), np.max(x)
    xr = max(xmax - xmin, 1e-12)
    smax = max(dfb["Gene Count"]) * size_ratio    
    r_pt = np.sqrt(smax)                        
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_w_in = max(bbox.width, 1e-6)               
    pad = (r_pt / 72.0) / ax_w_in * xr * 1.5   
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.invert_yaxis()
    return sc


def plot_cluster_marker_heatmap(
    dataset_num, locs, groundtruth, CoxFormer,
    src_dir, save_dir,
    truth_column="ident.annot",
    topk=100,
    clusters=("Tumor", "Other"),
    cmap="PuOr",
    figsize=(4, 10),
    dpi=200,
    show_ticks=True,
    fontsize=3,
):
    adata_imp = sc.read_h5ad(os.path.join(save_dir, dataset_num, f"{dataset_num}_CoxFormer_deg.h5ad"))

    s = adata_imp.obs[truth_column].astype(str)
    adata_imp.obs["region"] = pd.Categorical(
        np.where(s.eq("Tumor"), "Tumor", "Other"),
        categories=["Other", "Tumor"]
    )
    label = adata_imp.obs["region"].reset_index(drop=True)

    X = CoxFormer[adata_imp.var.index]
    markers, genes_show = get_topk_markers(X, label, topk=topk, clusters=list(clusters))
    ordered_spots = order_spots_by_marker_score(X, label, markers, list(clusters))

    hm = CoxFormer.loc[ordered_spots, genes_show]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    vmin, vmax = plot_gaussian_heatmap(
        ax, hm, vmin=None, vmax=None, cmap=cmap,
        show_ticks=show_ticks, fontsize=fontsize
    )
    plt.tight_layout()
    file_path = os.path.join(save_dir, dataset_num, f"{dataset_num}_cluster_marker.pdf")
    if not os.path.exists(file_path):
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    else:
        print(f"{file_path} already exists, skip saving.")
    plt.show()
    return