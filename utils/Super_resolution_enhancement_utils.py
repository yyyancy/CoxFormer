import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad
import liana as li
from liana.resource import select_resource

from scipy import stats
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

from plotnine import (
    theme, element_text, element_blank,
    scale_colour_gradientn, coord_flip,
    element_rect, guides, guide_legend, guide_colorbar
)

from PIL import Image

def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

# ------------------------ metrics ------------------------
def cal_ssim_1d(im1, im2, M):
    assert im1.ndim == 2 and im2.ndim == 2 and im1.shape == im2.shape
    mu1, mu2 = im1.mean(), im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()

    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return l12 * c12 * s12


def compute_metrics(pred, true, fillna_val=1e-20):
    true = np.asarray(true).reshape(-1).astype(float)
    pred = np.asarray(pred).reshape(-1).astype(float)

    true = np.nan_to_num(true, nan=fillna_val, posinf=fillna_val, neginf=fillna_val)
    pred = np.nan_to_num(pred, nan=fillna_val, posinf=fillna_val, neginf=fillna_val)

    # RMSE in z-score space
    true_z = stats.zscore(true, nan_policy="omit")
    pred_z = stats.zscore(pred, nan_policy="omit")
    true_z = np.nan_to_num(true_z, nan=0.0, posinf=0.0, neginf=0.0)
    pred_z = np.nan_to_num(pred_z, nan=0.0, posinf=0.0, neginf=0.0)

    rmse = np.sqrt(np.mean((true_z - pred_z) ** 2))

    # SSIM in raw space (vector -> column)
    true_col = true.reshape(-1, 1)
    pred_col = pred.reshape(-1, 1)
    M = float(max(true_col.max(), pred_col.max()))
    ssim_val = cal_ssim_1d(true_col, pred_col, M)

    # PCC (Pearson correlation); constant vector may yield NaN -> set 0
    pcc, _ = pearsonr(true.ravel(), pred.ravel())
    if np.isnan(pcc):
        pcc = 0.0

    return np.array([rmse, ssim_val, pcc], dtype=float)


# ------------------------ plot ------------------------
def plot_metrics_barh(metrics_matrix, method_names, metrics_labels, colors,
                      out_pdf_path=None, figsize=(11, 4.5), dpi=300):

    metrics_matrix = np.asarray(metrics_matrix, float)
    n_methods, n_metrics = metrics_matrix.shape

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    axes = np.array(axes).reshape(-1)
    y_pos = np.arange(n_methods)
    bar_h = 0.55

    cols = list(colors)
    if len(cols) < n_methods:
        cols = (cols * (n_methods // len(cols) + 1))[:n_methods]

    for i, ax in enumerate(axes):
        bars = ax.barh(
            y_pos,
            metrics_matrix[:, i],
            color=cols[:n_methods],
            alpha=1.0,
            height=bar_h,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(method_names) 
        for t in ax.get_yticklabels():
            t.set_rotation(90)
            t.set_va("center")           
            t.set_ha("center")           
            t.set_rotation_mode("anchor") 
            t.set_fontweight("bold")

        ax.tick_params(axis="y", pad=8, labelsize=18) 

        # x label
        ax.set_xlabel(metrics_labels[i], fontweight="bold", fontsize=18)
        ax.set_ylim(-0.45, n_methods - 1 + 0.45)

        # x/y tick style
        ax.tick_params(axis="x", labelsize=18)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")

        # spine style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_linewidth(2)

        for bar in bars:
            bar.set_edgecolor("black")
            bar.set_linewidth(2)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.20, wspace=0.25)
    if not os.path.exists(out_pdf_path):
        plt.savefig(out_pdf_path,  bbox_inches="tight")
    else:
        print(f"{out_pdf_path} already exists, skip saving.")

    plt.show()
    return fig

def evaluate_and_plot(
    save_dir: str,
    data: dict,
    metrics_labels=("RMSE", "SSIM", "PCC"),
    colors=("#DBE0ED", "#CC88B0"),
    out_pdf_name="bar_plot.pdf",
    align_to_shortest=True,
):
    # load gt
    gt_path = os.path.join(save_dir, data["gt"])
    y_true = np.load(gt_path).reshape(-1)

    # load preds
    method_names = list(data["pred"].keys())
    preds = []
    for m in method_names:
        pred_path = os.path.join(save_dir, data["pred"][m])
        preds.append(np.load(pred_path).reshape(-1))

    # align length
    if align_to_shortest:
        n = min([len(y_true)] + [len(p) for p in preds])
        y_true = y_true[:n]
        preds = [p[:n] for p in preds]

    # compute metrics: [RMSE, SSIM, PCC]
    metrics_matrix = np.vstack([compute_metrics(p, y_true) for p in preds])

    # plot
    out_pdf_path = os.path.join(save_dir, out_pdf_name)
    plot_metrics_barh(
        metrics_matrix=metrics_matrix,
        method_names=method_names,
        metrics_labels=list(metrics_labels),
        colors=colors,
        out_pdf_path=out_pdf_path,
    )
    return metrics_matrix



# ------------------------- basic utils -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def block_downsample_sum(arr, block_size):
    if block_size == 1:
        return arr
    else:
        H, W = arr.shape
        Hc = (H // block_size) * block_size
        Wc = (W // block_size) * block_size
        arr_c = arr[:Hc, :Wc]
        # reshape成块并求sum
        arr_ds = arr_c.reshape(Hc // block_size, block_size, Wc // block_size, block_size).sum(axis=(1, 3))
    return arr_ds

def normalize01_nan(x, eps=1e-12):
    x = x.astype(float, copy=False)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    x = x - mn
    x = x / (mx + eps)
    return x

def to_rgb_turbo(x01, mask_bool, clip_max=0.6):
    x01 = np.clip(x01, 0, 1)
    x_vis = x01 / clip_max
    cmap = plt.get_cmap("turbo")
    rgb = cmap(x_vis)[..., :3]  # float 0~1
    rgb[~mask_bool] = 1.0       
    rgb = (rgb * 255).astype(np.uint8)
    return rgb

# ------------------------- alignment config -------------------------
class XeniumAlignConfig:
    def __init__(
        self,
        pixel_size_raw=0.27377345896586774,
        pixel_size=0.5,
        start_y=0,
        start_x=0,
        crop_h=704,
        crop_w=784,
        base=975,
        div=16,
        panel_extra_shift=35, 
    ):
        self.pixel_size_raw = pixel_size_raw
        self.pixel_size = pixel_size
        self.start_y = start_y
        self.start_x = start_x
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.base = base
        self.div = div
        self.panel_extra_shift = panel_extra_shift

    @property
    def scale(self):
        return self.pixel_size_raw / self.pixel_size

    @property
    def w_start(self):
        # 你第二段：w_start = int(975*scale/16)
        return int(self.base * self.scale / self.div)

    @property
    def margin(self):
        # 你第一段：margin = np.int32(975*scale/16)
        return np.int32(self.base * self.scale / self.div)


def build_masks(cfg: XeniumAlignConfig, mask_path="mask-cell.png"):
    mask_origin = load_image(mask_path) > 0
    y0, x0 = cfg.start_y, cfg.start_x
    H, W = cfg.crop_h, cfg.crop_w
    m = cfg.margin
    mask_gt = mask_origin[y0:y0 + H, x0 + m:x0 + m + W]
    mask_pred = mask_gt.copy()
    return mask_gt, mask_pred


def crop_and_align_gt(cnts_gt, H, W, x_offset):
    cnts_gt = cnts_gt[:H, x_offset:x_offset + W]
    return cnts_gt

# ------------------------- main plotting function -------------------------
def plot_xenium_gene_show(
    gene_show,
    sources,                 # dict: {"GT": dict/np arrays, "CoxFormer": dict, "iStar": dict (optional)}
    cfg: XeniumAlignConfig,
    mask_path="mask-cell.png",
    block_size=1,
    clip_max=1,
    out_dir=None,
    out_prefix="Hyper",
    dpi=300,
    figsize_per_col=4.5,
    spine_lw=2,
    show=True,
):
    """
    gene_show: list[str]
    sources:
      - sources["GT"]   : dict-like, gene -> 2D array
      - sources["CoxFormer"] : dict-like, gene -> 2D array
      - sources["iStar"] (optional): dict-like, gene -> 2D array
    """
    assert "GT" in sources and "CoxFormer" in sources, "sources at least need 'GT' and 'CoxFormer'"
    has_istar = ("iStar" in sources) and (sources["iStar"] is not None)

    # masks
    mask_gt, mask_pred = build_masks(cfg, mask_path=mask_path)

    # x offset for GT alignment
    x_offset = cfg.margin

    # output
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    col_names = ["GT", "CoxFormer"] + (["iStar"] if has_istar else [])
    ncols = len(col_names)

    for gn in gene_show:
        # ---- load arrays ----
        gt = sources["GT"][gn]
        CoxFormer = sources["CoxFormer"][gn]
        istar = sources["iStar"][gn] if has_istar else None

        # ---- crop preds (start_y/start_x) ----
        y0, x0 = cfg.start_y, cfg.start_x
        CoxFormer = CoxFormer[y0:, x0:]
        H, W = CoxFormer.shape

        if has_istar:
            istar = istar[y0:, x0:]
            H = min(H, istar.shape[0])
            W = min(W, istar.shape[1])
            CoxFormer = CoxFormer[:H, :W]
            istar = istar[:H, :W]
        else:
            CoxFormer = CoxFormer[:H, :W]

        # ---- crop GT with x_offset ----
        gt = crop_and_align_gt(gt, H, W, x_offset=x_offset)

        # ---- downsample ----
        gt_ds = block_downsample_sum(gt, block_size)
        CoxFormer_ds = block_downsample_sum(CoxFormer, block_size)
        if has_istar:
            istar_ds = block_downsample_sum(istar, block_size)

        # ---- downsample masks ----
        mask_gt_ds = block_downsample_sum(mask_gt.astype(float), block_size) > 0
        mask_pred_ds = block_downsample_sum(mask_pred.astype(float), block_size) > 0
        mask_vis = mask_pred_ds

        # ---- normalize + to rgb ----
        gt_vis = gt_ds.astype(float)
        gt_vis[~mask_vis] = np.nan
        gt_01 = normalize01_nan(gt_vis)
        img_gt = to_rgb_turbo(gt_01, mask_vis, clip_max=clip_max)

        # CoxFormer
        CoxFormer_vis = CoxFormer_ds.astype(float)
        CoxFormer_vis[~mask_vis] = np.nan
        CoxFormer_01 = normalize01_nan(CoxFormer_vis)
        img_CoxFormer = to_rgb_turbo(CoxFormer_01, mask_vis, clip_max=clip_max)

        # iStar
        if has_istar:
            istar_vis = istar_ds.astype(float)
            istar_vis[~mask_vis] = np.nan
            istar_01 = normalize01_nan(istar_vis)
            img_istar = to_rgb_turbo(istar_01, mask_vis, clip_max=clip_max)

        # ---- plot ----
        fig_w = figsize_per_col * ncols
        fig, ax = plt.subplots(1, ncols, figsize=(fig_w, 4), dpi=dpi)

        if ncols == 1:
            ax = [ax]

        imgs = [img_gt, img_CoxFormer] + ([img_istar] if has_istar else [])
        for j in range(ncols):
            ax[j].imshow(imgs[j])
            ax[j].set_xticks([])
            ax[j].set_yticks([])
            for sp in ax[j].spines.values():
                sp.set_linewidth(spine_lw)

        plt.tight_layout()

        if out_dir is not None:
            out_path = os.path.join(out_dir, f"{out_prefix}_{gn}.pdf")
            plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)

        if show:
            plt.show()
        else:
            plt.close(fig)



def plot_spatial_clustering(
    pred_pkl_path: str,            # .../CoxFormer-Img_pixel_sim_hyper.pkl 或 .../iStar_pixel_sim_hyper.pkl
    mask_png_path: str,            # .../mask-cell-sub.png   
    out_pdf_path: str,            
    n_clusters: int = 3,
    n_pcs: int = 10,
    colors=("#CC88B0", "#F4CEB4", "#9C8CBB"),
    random_state: int = 42,
):
    """
    读 pkl -> normalize -> dict转AnnData -> PCA -> KMeans -> 画空间聚类图并保存
    """
    mask = load_image(mask_png_path) > 0

    with open(pred_pkl_path, "rb") as f:
        gene_dict = pickle.load(f)
    gene_dict = normalize_dict(gene_dict)
    adata = convert_dict_adata(gene_dict, mask)

    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    Xpc = adata.obsm["X_pca"]

    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(Xpc)
    adata.obs["kmeans"] = pd.Categorical(labels.astype(str))

    cats = list(adata.obs["kmeans"].cat.categories)
    assert len(colors) >= len(cats), f"colors at least need {len(cats)}"

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    for i, k in enumerate(cats):
        idx = adata.obs["kmeans"] == k
        x = adata.obs.loc[idx, "x"].values
        y = adata.obs.loc[idx, "y"].values

        # 你原来的旋转
        ax.scatter(
            y, -x,
            s=1,
            marker="o",
            color=colors[i],
            edgecolors="none",
            alpha=1.0,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0)
    plt.subplots_adjust(0, 0, 1, 1)

    os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)
    plt.savefig(out_pdf_path, bbox_inches="tight")
    plt.show()
    return adata
    
def convert_dict_adata(gene_dict,mask):
    genes = list(gene_dict.keys())
    h, w = next(iter(gene_dict.values())).shape
    if mask.dtype != bool:
        mask = mask.astype(bool)
    mask_flat = mask.reshape(-1)             # (h*w,)
    n_spot = int(mask_flat.sum())
    n_gene = len(genes)
    X = np.zeros((n_spot, n_gene), dtype=np.float32)
    for j, g in enumerate(genes):
        X[:, j] = gene_dict[g].reshape(-1)[mask_flat]
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))  
    coords_all = np.column_stack([xx.reshape(-1), yy.reshape(-1)])  # (h*w, 2) -> [x, y]
    coords_kept = coords_all[mask_flat]
    coords = pd.DataFrame(coords_kept, columns=["x", "y"])
    coords["orig_index"] = np.flatnonzero(mask_flat)
    var = pd.DataFrame(index=genes)
    adata = ad.AnnData(X=X, obs=coords, var=var)
    return adata

def normalize_dict(gene_dict):
    gene_dict_norm = {}
    for k, v in gene_dict.items():
        arr = v.astype(float)  
        v_min, v_max = arr.min(), arr.max()
        if v_max > v_min: 
            arr_norm = (arr - v_min) / (v_max - v_min)
        else:
            arr_norm = np.zeros_like(arr) 
        gene_dict_norm[k] = arr_norm
    return gene_dict_norm


def run_liana_ccc_spatial(
    adata,
    celltype_csv,
    coords_csv,
    celltype_obs_key="cell_type",
    # celltype csv columns
    barcode_col="Barcode",
    cluster_col="Cluster",
    # coords csv columns
    coord_index_col="cell_id",
    x_col="x_centroid",
    y_col="y_centroid",
    # rank_aggregate
    resource_name="consensus",
    expr_prop=0.1,
    use_raw=False,
    verbose=True,
    # spatial neighbors
    bandwidth=100,
    cutoff=0.1,
    kernel="gaussian",
    set_diag=True,
    # bivariate
    local_name="cosine",
    global_name="morans",
    n_perms=100,
    mask_negatives=False,
    add_categories=True,
    nz_prop=0.2,
    # plotting
    make_spatial_plots=True,
    spot_size=15,
    pvals_cmap="magma_r",
    cats_cmap="coolwarm",
    scanpy_figsize=(5, 5),
    out_h5ad_path="Result/Super_resolution_enhancement/skin/cell_cell_communication/lrdata_GT.h5ad",
):
    # 1) cell types
    clusters = pd.read_csv(celltype_csv).set_index(barcode_col)
    adata.obs[celltype_obs_key] = (
        clusters[cluster_col].reindex(adata.obs.index).fillna("unsigned").astype(str).astype("category")
    )

    # 2) rank_aggregate
    resource = select_resource(resource_name)
    li.mt.rank_aggregate(
        adata,
        groupby=celltype_obs_key,
        resource=resource,
        expr_prop=expr_prop,
        use_raw=use_raw,
        verbose=verbose,
    )
    liana_res = adata.uns["liana_res"]

    # 3) LR candidates for bivariate
    lr_pairs = (
        liana_res[["ligand_complex", "receptor_complex"]]
        .drop_duplicates()
        .rename(columns={"ligand_complex": "ligand", "receptor_complex": "receptor"})
    )

    # 4) spatial coords
    cells = pd.read_csv(coords_csv, compression="infer").set_index(coord_index_col)
    xy = cells.loc[adata.obs.index, [x_col, y_col]].copy()
    adata.obs[[x_col, y_col]] = xy.values
    adata.obsm["spatial"] = xy.to_numpy()

    # 5) spatial neighbors
    li.ut.spatial_neighbors(
        adata,
        bandwidth=bandwidth,
        cutoff=cutoff,
        kernel=kernel,
        set_diag=set_diag,
    )

    # 6) bivariate
    lrdata = li.mt.bivariate(
        adata,
        resource=lr_pairs,
        local_name=local_name,
        global_name=global_name,
        n_perms=n_perms,
        mask_negatives=mask_negatives,
        add_categories=add_categories,
        nz_prop=nz_prop,
        use_raw=use_raw,
        verbose=verbose,
    )

    # 7) quick spatial plots
    if make_spatial_plots:
        sc.set_figure_params(format="png", frameon=False, transparent=True, figsize=list(scanpy_figsize))
        lr_top = lrdata.var.sort_values("morans", ascending=False).head(8).index.tolist()

        sc.pl.spatial(lrdata, color=lr_top, spot_size=spot_size)
        sc.pl.spatial(lrdata, layer="pvals", color=lr_top, spot_size=spot_size, cmap=pvals_cmap)
        sc.pl.spatial(lrdata, layer="cats",  color=lr_top, spot_size=spot_size, cmap=cats_cmap)

    if not os.path.exists(out_h5ad_path):
        lrdata.write_h5ad(out_h5ad_path, compression="gzip")
    return lrdata

def plot_liana_lr_dotplot(lrdata_h5ad_path: str, method: str, out_dir: str = "Result/Super_resolution_enhancement/skin/cell_cell_communication/"):
    adata = sc.read_h5ad(lrdata_h5ad_path)

    mapping = {
        "1": "Malanoma Tumor", "2": "Immune Infiltration", "3": "Epidermis",
        1: "Malanoma Tumor",  2: "Immune Infiltration",  3: "Epidermis",
    }

    res = adata.uns["liana_res"].copy()
    res["source"] = res["source"].map(mapping).fillna(res["source"]).astype(str)
    res["target"] = res["target"].map(mapping).fillna(res["target"]).astype(str)
    adata.uns["liana_res"] = res

    res = adata.uns["liana_res"].copy()
    res["lr"] = res.apply(lambda r: f"{r['ligand_complex']} -> {r['receptor_complex']}", axis=1)
    res["bi"] = res.apply(lambda r: f"{r['ligand_complex']}^{r['receptor_complex']}", axis=1)

    keep = set(map(str, adata.var_names))
    res = res[res["bi"].isin(keep)]

    groups = ["Epidermis", "Immune Infiltration", "Malanoma Tumor"]
    top_set = set()
    for g in groups:
        tmp = res[(res["source"] == g) | (res["target"] == g)].sort_values("specificity_rank").head(10)
        top_set.update(tmp["lr"].tolist())

    p = li.pl.dotplot(
        adata=adata,
        colour="lrscore",
        inverse_colour=False,
        size="specificity_rank",
        inverse_size=True,
        filter_fun=lambda r: f"{r['ligand_complex']} -> {r['receptor_complex']}" in top_set,
    )

    p = (
        p
        + guides(
            size=guide_legend(title="Specificity Rank"),
            colour=guide_colorbar(title="LR Score"),
        )
        + theme(
            panel_background=element_rect(fill="white", colour=None),
            plot_background=element_rect(fill="white", colour=None),
            axis_title_x=element_blank(),
            axis_title_y=element_blank(),
            plot_title=element_blank(),
            axis_text_x=element_text(rotation=90, ha="right", size=10, weight="normal", color="black"),
            strip_text=element_text(size=10, weight="bold",  color="black"),
            axis_text_y=element_text(rotation=45, size=10, weight="bold", color="black"),
            legend_title=element_text(size=10, weight="normal", color="black"),
            legend_text=element_text(size=10, weight="normal", color="black"),
            figure_size=(18, 5.5),
            panel_border=element_rect(colour="black", size=2, fill=None),
            strip_background=element_rect(colour="black", size=2, fill="white"),
        )
        + scale_colour_gradientn(colors=["#552E81", "#992F87", "#EFAE42"])
        + coord_flip()
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lr_dotplot_{method}.pdf")
    if not os.path.exists(out_path):
        p.save(out_path, dpi=300, width=18, height=5.5, limitsize=False)
    return p

def plot_liana_nmf_spatial(lrdata_h5ad_path: str, method: str, save_path: str, raw_save=True,
                           n_components: int = 3, spot_size: int = 10):
    lrdata = sc.read_h5ad(lrdata_h5ad_path)
    li.multi.nmf(lrdata, n_components=n_components, inplace=True,
                 random_state=0, max_iter=1000, verbose=True)

    lr_loadings = li.ut.get_variable_loadings(lrdata, varm_key="NMF_H").set_index("index")
    nmf_base = sc.AnnData(
        X=lrdata.obsm["NMF_W"],
        obs=lrdata.obs,
        var=pd.DataFrame(index=lr_loadings.columns),
        uns=lrdata.uns,
        obsm=lrdata.obsm
    )

    sc.pl.spatial(
        nmf_base,
        color=list(nmf_base.var.index),
        spot_size=spot_size,
        ncols=n_components,
        show=False,
        frameon=False,
    )

    if not raw_save:
        fig = plt.gcf()
        for ax in fig.axes:
            ax.set_title(ax.get_title(), fontsize=14, fontweight="normal")
    out_path = os.path.join(save_path, f"nmf_spatial_{method}.png")
    if not os.path.exists(out_path):
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    return


from torch import nn
class FeedForward(nn.Module):
    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class ForwardSumModel(nn.Module):
    def __init__(self, n_inp, n_out):
        super().__init__()
        self.net_lat = nn.Sequential(
                FeedForward(n_inp, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        self.net_out = FeedForward(
                256, n_out,
                activation=ELU(alpha=0.01, beta=0.01))

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x
