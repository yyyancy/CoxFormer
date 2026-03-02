##import packages
# ====== Basic Python Utilities ======
import os               # OS path, file operations, environment variables

# ====== Numerical Computing & Data Processing ======
import numpy as np                   # Numerical computing (arrays, matrix ops)
import pandas as pd                 # DataFrame for tabular data manipulation

# ====== Machine Learning / Data Preprocessing ======
from sklearn.metrics import mean_squared_error, r2_score        # Evaluation metrics

# ====== Bioinformatics (Single-cell / Spatial Data) ======
import scanpy as sc       # Core single-cell / spatial transcriptomics framework

# ====== Visualization ======
import matplotlib.pyplot as plt        # Plotting with matplotlib
import seaborn as sns                  # Statistical visualization (heatmap, KDE, etc.)

# ====== Scientific Computing ======
from scipy.ndimage import gaussian_filter1d   # 1D Gaussian smoothing

# ====== Visualization ======
import matplotlib.pyplot as plt               # Plotting with matplotlib
from utils.Gene_expression_prediction_utils import save_colorbar, plot_spatial_scatter

def plot_radar(PATH, metrics, Tools, colors, save):
    result = pd.DataFrame()
    for tool in Tools:
        df = pd.read_csv(f"{PATH}/{tool}_Metrics.txt", sep="\t", header=0, index_col=0)
        df["tool"] = tool
        result = pd.concat([result, df], axis=0)

    mean_values = result.groupby("tool")[metrics].mean()
    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),dpi=300)
    for i, tool in enumerate(Tools):
        values = mean_values.loc[tool].tolist()
        values += values[:1]
        values[2] = 1.5 - values[2]
        ax.plot(angles, values, label=tool, linewidth=2, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.5, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([''] * num_vars)
    #ax.set_xticklabels(labels, fontsize=12, fontweight="bold")

    # 美化
    ax.set_rlabel_position(90)
    ax.tick_params(axis='y', labelsize=7)

    # 保存
    if save == True:
        plt.savefig(f"{PATH}/radar.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close()
    
    
def plot_bars_by_metric(PATH, metrics, Tools, colors, save,
                        y_min=None, y_max=None, margin=0.05):
    # ---- rename for display ----
    name_map = {
        "CoxFormer-Loc": "CoxFormer",
        "coexpression_pca-Loc": "Gene Coexpression",
        "correlation_pca-Loc": "Gene Correlation",
        "text-Loc": "Gene Description",
    }
    Tools_disp = [name_map.get(t, t) for t in Tools]

    # ---- load & concat ----
    result = []
    for tool in Tools:
        f = os.path.join(PATH, f"{tool}_Metrics.txt")
        df = pd.read_csv(f, sep="\t", header=0, index_col=0)
        df["tool"] = tool
        result.append(df)
    result = pd.concat(result, axis=0)

    # ---- mean by tool ----
    mean_values = result.groupby("tool")[metrics].mean()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 3.2), dpi=300)
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(Tools))

    for j, m in enumerate(metrics):
        ax = axes[j]
        y = mean_values.loc[Tools, m].values  # 按 Tools 顺序排列

        ax.bar(
            x, y,
            color=[colors[i % len(colors)] for i in range(len(Tools))],
            edgecolor="black",
            linewidth=0.6
        )

        # ---- y-limits ----
        y_min = np.nanmin(y)
        y_max = np.nanmax(y)
        ax.set_ylim(max(0,y_min-0.05), y_max+0.05)

        ax.set_title(m, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(Tools_disp, rotation=45, ha="right", fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()

    if save == True:
        plt.savefig(os.path.join(PATH, "bars.pdf"), bbox_inches="tight", pad_inches=0.05)

    plt.show()
    plt.close(fig)


def run_deg_wilcoxon(gt_df, labels, target_label):
    spots_t = labels.index[labels == target_label]
    spots_r = labels.index[labels != target_label]

    target = sc.AnnData(
        X=gt_df.loc[spots_t].to_numpy(),
        obs=pd.DataFrame(index=spots_t),
        var=pd.DataFrame(index=gt_df.columns),
    )
    ref = sc.AnnData(
        X=gt_df.loc[spots_r].to_numpy(),
        obs=pd.DataFrame(index=spots_r),
        var=pd.DataFrame(index=gt_df.columns),
    )
    combined = sc.concat([target, ref], label="group", keys=["target", "ref"], merge="same")
    sc.tl.rank_genes_groups(combined, groupby="group", method="wilcoxon")
    deg_df = sc.get.rank_genes_groups_df(combined, group="target")
    return deg_df


def plot_gaussian_heatmap(ax, df, vmin, vmax, cmap, show_ticks=True,fontsize=14):
    X = df.to_numpy().T
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    X = (X - mu) / (sd + 1e-9)
    if vmin == None:
        vmin, vmax = np.percentile(X, [5, 95])
    X = gaussian_filter1d(X, sigma=3, axis=1, mode="nearest")
    im = ax.imshow(X, aspect="auto", interpolation="nearest",cmap=cmap,vmin=vmin, vmax=vmax)
    if show_ticks:
        ax.set_yticks(np.arange(len(df.columns)))
        ax.set_yticklabels(df.columns, fontsize=fontsize, fontname="Nimbus Sans", fontweight="bold")
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    return vmin, vmax

def plot_atac_spatial(
    gene_list,
    location,
    data,         
    save_dir=None,
    methods=("GroundTruth", "CoxFormer", "GenePT"),
    cmap="coolwarm_r",
    s=10,
    add_colorbar=False,
    dpi=300,
    spine_lw=3,
    pad_inches=0.1,
):
    os.makedirs(save_dir, exist_ok=True)
    figsize = (3, 7 / 3 * len(methods))
    for g in gene_list:
        print(f"Gene: {g}")
        fig, axes = plt.subplots(len(methods), 1, figsize=figsize, dpi=dpi)
        axes = [axes] if len(methods) == 1 else axes

        for i, m in enumerate(methods):
            if m not in data or data[m] is None or g not in data[m].columns:
                axes[i].text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
                axes[i].axis("off")
                continue
            expr = data[m][g]
            plot_spatial_scatter(
                axes[i],
                location,
                expr,
                mode="continuous",
                cmap=cmap,
                add_colorbar=add_colorbar,
                s=s,
            )

            for sp in axes[i].spines.values():
                sp.set_visible(True)
                sp.set_linewidth(spine_lw)

        plt.tight_layout()
        if save_dir:
            file_path = os.path.join(save_dir, f"plot_{g}.pdf")
            if not os.path.exists(file_path):
                plt.savefig(file_path, bbox_inches="tight", pad_inches=pad_inches)
            else:
                print(f"{file_path} already exists, skip saving.")
        plt.show()
        plt.close(fig)


def align_common_genes(gt, *mats):
    common = gt.columns
    for m in mats:
        common = common.intersection(m.columns)
    common = common.astype(str)  # 可选：防止列名类型不一致

    out = [gt[common].copy()]
    out += [m[common].copy() for m in mats]
    return out  # 返回 list： [gt_aligned, mat1_aligned, mat2_aligned, ...]

def subset_spots_by_group(meta, gt, *mats):
    meta = meta.copy()
    meta["group3"] = meta["ATAC_clusters"].replace({"C1": "CA3", "C4": "GCL"})
    meta.loc[~meta["group3"].isin(["CA3", "GCL"]), "group3"] = np.nan

    keep = meta.index[meta["group3"].notna()].intersection(gt.index)
    labels = meta.loc[keep, "group3"].astype(str)

    gt3 = gt.loc[keep].copy()
    mats3 = [m.loc[keep].copy() for m in mats]
    return [gt3, *mats3, labels]


def get_topk_markers(gt, labels, topk=7, clusters=("CA3", "GCL")):
    # 1) collect candidates with scores
    cand_rows = []
    for c in clusters:
        deg_df = run_deg_wilcoxon(gt, labels, c)
        sub = deg_df.sort_values("scores", ascending=False)[["names", "scores"]].head(topk).copy()
        sub["names"] = sub["names"].astype(str)
        sub["cluster"] = c
        sub = sub[sub["names"].isin(gt.columns)]
        cand_rows.append(sub)

    cand = pd.concat(cand_rows, ignore_index=True)  # columns: names, scores, cluster
    # 2) for each gene, keep the cluster with higher score (tie -> first seen)
    cand = cand.sort_values(["names", "scores"], ascending=[True, False])
    best = cand.drop_duplicates(subset=["names"], keep="first")  # unique genes

    # 3) assign back to clusters, keep per-cluster order by score
    markers = {}
    genes_show = []
    for c in clusters:
        g = best[best["cluster"] == c].sort_values("scores", ascending=False)["names"].tolist()
        markers[c] = g
        genes_show.extend(g)
    return markers, genes_show

def order_spots_by_marker_score(gt, labels, markers, clusters = ["CA3", "GCL"]):
    spot_order = gt.index
    ordered_spots = []
    boundaries = []
    for c in clusters:
        spots_c = spot_order[labels.loc[spot_order] == c]
        g_c = markers.get(c, [])

        if len(g_c) == 0:
            spots_sorted = list(spots_c)
        else:
            score = gt.loc[spots_c, g_c].mean(axis=1)
            spots_sorted = score.sort_values(ascending=False).index.tolist()

        ordered_spots.extend(spots_sorted)
        boundaries.append(len(ordered_spots))
    return ordered_spots


def plot_cluster_heatmaps(
    data,            
    meta,
    clusters=("CA3", "GCL"),
    topk=7,
    RES_PATH=".",
    cmap="RdBu_r",
    dpi=200,
    base_w=6,
    gt_w=7,
    pad_inches=0.1,
    save=False,
):
    os.makedirs(RES_PATH, exist_ok=True)
    assert "GroundTruth" in data, "data 里必须包含 'GroundTruth'"

    # ---- 0) pick GroundTruth + other methods in order ----
    gt = data["GroundTruth"]
    method_names = [k for k in data.keys() if k != "GroundTruth"]
    mats = [data[k] for k in method_names]

    # ---- 1) align genes once (gt + all methods) ----
    gt, *mats = align_common_genes(gt, *mats)

    # ---- 2) subset spots by group (gt + all methods) ----
    gt, *mats, labels = subset_spots_by_group(meta, gt, *mats)

    # ---- 3) DEG markers (topK per cluster)  (沿用你原逻辑：用第一个方法算markers) ----
    markers, genes_show = get_topk_markers(mats[0], labels, topk=topk, clusters=list(clusters))

    # ---- 4) order spots within each cluster by GT marker score ----
    ordered_spots = order_spots_by_marker_score(gt, labels, markers, list(clusters))

    # ---- 5) build heatmap matrices ----
    gt_hm = gt.loc[ordered_spots, genes_show]
    hm_dict = {name: mat.loc[ordered_spots, genes_show] for name, mat in zip(method_names, mats)}

    # ---- 6) plot GroundTruth first to get vmin/vmax ----
    print("Ground Truth:")
    fig, ax = plt.subplots(1, 1, figsize=(gt_w, max(4, 0.2 * len(genes_show))), dpi=dpi)
    vmin, vmax = plot_gaussian_heatmap(ax, gt_hm, vmin=None, vmax=None, cmap=cmap, show_ticks=True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(RES_PATH, "cluster_GroundTruth.pdf"), bbox_inches="tight", pad_inches=pad_inches)
    plt.show()
    plt.close(fig)

    # ---- 7) plot all other methods (auto) ----
    for name, mat in hm_dict.items():
        print(f"{name}:")
        fig, ax = plt.subplots(1, 1, figsize=(base_w, max(4, 0.2 * len(genes_show))), dpi=dpi)
        plot_gaussian_heatmap(ax, mat, vmin=vmin, vmax=vmax, cmap=cmap, show_ticks=False)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(RES_PATH, f"cluster_{name}.pdf"), bbox_inches="tight", pad_inches=pad_inches)
        plt.show()
        plt.close(fig)

    # ---- 8) save shared colorbar ----
    save_colorbar(vmin, vmax, cmap, RES_PATH, figsize=(0.5, max(4, 0.2 * len(genes_show))))
    return
