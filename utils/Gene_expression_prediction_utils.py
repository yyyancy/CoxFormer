##import packages
# ====== Basic Python Utilities ======
import os               # OS path, file operations, environment variables

# ====== Numerical Computing & Data Processing ======
import numpy as np                   # Numerical computing (arrays, matrix ops)
import pandas as pd                 # DataFrame for tabular data manipulation
import scipy.stats as st            # Statistical functions
from scipy import stats             # Additional statistical tools
from scipy.stats import (           # Specific statistical metrics
    pearsonr, spearmanr
)
import math

# ====== Machine Learning / Data Preprocessing ======
from sklearn.metrics import mean_squared_error, r2_score        # Evaluation metrics


# ====== Visualization ======
import matplotlib.pyplot as plt               # Plotting with matplotlib
from matplotlib.lines import Line2D           # Custom legend handles
from matplotlib.ticker import FormatStrFormatter  # Tick label formatting
import matplotlib.cm as cm                    # Colormaps
import matplotlib.colors as mcolors           # Color utilities
import seaborn as sns                  # Statistical visualization (heatmap, KDE, etc.)


def cal_ssim(im1,im2,M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim

def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.max()
        result = pd.concat([result, content],axis=1)
    return result

def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content,columns=[label])
        result = pd.concat([result, content],axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content/content.sum()
        result = pd.concat([result,content],axis=1)
    return result

def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df
    
class CalculateMeteics:
    def __init__(self, raw_count_file, impute_count_file, prefix, metric, genes):            
        self.raw_count = pd.read_csv(raw_count_file, header = 0, sep=None)        
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.impute_count = pd.read_csv(impute_count_file, header = 0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        if genes == None:
            genes = sorted(set(self.raw_count.columns) & set(self.impute_count.columns))
        else:
            genes = [g.upper() for g in genes]
        
        self.raw_count = self.raw_count[genes] 
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)
        #self.raw_count = self.raw_count.T

        self.impute_count_file = impute_count_file

        self.impute_count = self.impute_count[genes]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric
        #self.impute_count = self.impute_count.T
        
    def SSIM(self, raw, impute, scale = 'scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print ('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(),impute_col.max()][raw_col.max()>impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                    ssim = cal_ssim(raw_col_2,impute_col_2,M)
                
                ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
                result = pd.concat([result, ssim_df],axis=1)
        else:
            print("columns error")
        return result
            
    def PCC(self, raw, impute, scale = None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    pearsonr = 0
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    pearsonr, _ = st.pearsonr(raw_col,impute_col)
                    if pd.isna(pearsonr):
                        pearsonr = 0
                pearson_df = pd.DataFrame(pearsonr, index=["PCC"],columns=[label])
                result = pd.concat([result, pearson_df],axis=1)
        else:
            print("columns error")
        return result
    
    def RMSE(self, raw, impute, scale = 'zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print ('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5   
                else:
                    raw_col =  raw.loc[:,label]
                    impute_col = impute.loc[:,label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
                result = pd.concat([result, RMSE_df],axis=1)
        else:
            print("columns error")
        return result       
        
    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM = self.SSIM(raw,impute)
        Pearson = self.PCC(raw, impute)
        RMSE = self.RMSE(raw, impute)
        
        result_all = pd.concat([Pearson, SSIM, RMSE],axis=0)
        result_all.T.to_csv(prefix + "_Metrics.txt", sep='\t', header = 1, index = 1)
        self.accuracy = result_all
        return result_all


def CalDataMetric(PATH,CountInsuteDir,gene_list=None):
    metric = ['PCC','SSIM','RMSE']
    impute_count_dir = PATH
    impute_count = os.listdir(impute_count_dir)
    impute_count = [
                        x for x in impute_count
                        if x.endswith(".csv") and not x.startswith("._") and x[:-4].endswith("impute")
                    ]
    if len(impute_count)!=0:
        for impute_count_file in impute_count:
            prefix = impute_count_file[:-len("_impute.csv")]
            prefix = impute_count_dir + '/' + prefix
            impute_count_file = impute_count_dir + '/' + impute_count_file
            if not os.path.isfile(prefix + '_Metrics.txt'):
                print(impute_count_file)
                CM = CalculateMeteics(raw_count_file = CountInsuteDir, impute_count_file = impute_count_file, prefix = prefix, metric = metric, genes = gene_list)
                CM.compute_all()


class MetricsCalculator:
    """Class for calculating various evaluation metrics for gene expression prediction."""
    
    @staticmethod
    def scale_max(df: pd.DataFrame) -> pd.DataFrame:
        """Scale data by maximum value to range [0,1]."""
        return df.div(df.max())
    
    @staticmethod
    def scale_plus(df: pd.DataFrame) -> pd.DataFrame:
        """Scale data by sum to range (0,1) with sum = 1."""
        return df.div(df.sum())
    
    @staticmethod
    def scale_z_score(df: pd.DataFrame) -> pd.DataFrame:
        """Scale data using Z-score normalization."""
        return pd.DataFrame(stats.zscore(df), columns=df.columns, index=df.index)
    
    @staticmethod
    def cal_ssim(im1: np.ndarray, im2: np.ndarray, M: float) -> float:
        """Calculate SSIM between two arrays."""
        mu1, mu2 = im1.mean(), im2.mean()
        sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
        sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
        sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
        
        k1, k2, L = 0.01, 0.03, M
        C1, C2 = (k1*L)**2, (k2*L)**2
        C3 = C2/2
        
        l12 = (2*mu1*mu2 + C1)/(mu1**2 + mu2**2 + C1)
        c12 = (2*sigma1*sigma2 + C2)/(sigma1**2 + sigma2**2 + C2)
        s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
        
        return l12 * c12 * s12
    
    def calculate_rmse(self, true_vals: pd.Series, pred_vals: pd.Series) -> float:
        """Calculate RMSE using z-scored values."""
        true_z = stats.zscore(true_vals)
        pred_z = stats.zscore(pred_vals)
        return np.sqrt(((true_z - pred_z) ** 2).mean())
    
    def calculate_pearson(self, true_vals: pd.Series, pred_vals: pd.Series) -> float:
        """Calculate Pearson correlation."""
        pearson, _ = st.pearsonr(true_vals, pred_vals)
        return pearson
        
    def calculate_ssim(self, true_vals: pd.Series, pred_vals: pd.Series) -> float:
        """Calculate SSIM."""
        true_max = self.scale_max(pd.DataFrame(true_vals)).values
        pred_max = self.scale_max(pd.DataFrame(pred_vals)).values
        M = max(true_max.max(), pred_max.max())
        return self.cal_ssim(true_max, pred_max, M)
    
    def compute_all_metrics(self, true_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        metrics = {
            'RMSE': [],
            'Pearson': [],
            'SSIM': []
        }
        
        for gene in true_df.columns:
            true_vals = true_df[gene]
            pred_vals = pred_df[gene]
            
            metrics['RMSE'].append(self.calculate_rmse(true_vals, pred_vals))
            metrics['Pearson'].append(self.calculate_pearson(true_vals, pred_vals))
            metrics['SSIM'].append(self.calculate_ssim(true_vals, pred_vals))
        
        return pd.DataFrame(metrics, index=true_df.columns)


def plot_bar_benchmark(PATHS,metric,Tools,outdir,save_name,ascending,save=False):
    methods = ['CoxFormer','CoxFormer-Loc','CoxFormer-Learn','CoxFormer-Img', "Tangram",'GenePT', "SpaGE",  "stPlus", "SpaOTsc", "gimVI",  "LIGER", "Seurat", "novoSpaRc"]
    colors  = ["#E56F5E","#F19685","#F1C89A","#FFB77F","#F6C957" ,"#FBE8D5", "#43978F", '#97CEBF', "#DCE9F4",'#8AB1D2',"#F8D5E4","#DBD8E9","#C2ABC8","#9DD0C7","#9EC4BE"] 
    palette_map = {m: c for m, c in zip(methods, colors)}
    tools_in_data = sorted(set(Tools))
    missing = [t for t in tools_in_data if t not in palette_map]
    if missing:
        extra_cols = sns.color_palette(n_colors=len(missing))  # 默认颜色
        for t, col in zip(missing, extra_cols):
            palette_map[t] = col
    
    result = pd.DataFrame()
    metric_values = []
    per_path_means = []  

    for path in PATHS: 
        for tool in Tools:
            df = pd.read_csv(f"{path}/{tool}_Metrics.txt", sep="\t", header=0, index_col=0)
            m = df[metric].mean()
            per_path_means.append({"tool": tool, metric: m, "path": path})
    
    result = pd.DataFrame(per_path_means)
    tool_avg = (result.groupby("tool", as_index=False)[metric]
                .mean()
                .sort_values(by=metric, ascending=ascending))
    sorted_tools = tool_avg["tool"].tolist()
    
    result["tool"] = pd.Categorical(result["tool"], categories=sorted_tools, ordered=True)    
    plt.figure(figsize=(5, 4),dpi=300)  
    ax = plt.gca()  
    sns.barplot(x=metric, y="tool", data=result, errorbar=('ci', 95),palette=palette_map, hue="tool", legend=False)
    special_tools = {}
    special_tools = {"CoxFormer","CoxFormer-Learn","CoxFormer-Loc","CoxFormer-Img"}

    tool_order = [t.get_text() for t in ax.get_yticklabels()]
    for i, tool in enumerate(tool_order):
        if tool in special_tools:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#E07B54",  alpha=0.2,  zorder=0)
        else:
            ax.axhspan(i - 0.5, i + 0.5, facecolor="#828D93", alpha=0.1, zorder=0)
    
    for patch, tool in zip(ax.patches, tool_order):
        patch.set_zorder(2)
        if tool in special_tools:
            patch.set_hatch('///')             
            patch.set_edgecolor('black')
            patch.set_linewidth(1)
        else:
            patch.set_edgecolor('black')
            patch.set_linewidth(1)

    x_min = result[metric].min()
    x_max = result[metric].max()
    n = len(tool_order)
    ax.margins(y=0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(left=max(0,x_min-0.2))  
    ax.set_xlim(right=x_max+0.1) 
    ax.set_ylabel("") 
    ax.set_xlabel(metric, fontweight='bold',fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=12)  
    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold') 
    
    # 保存和显示图形
    if save == True:
        plt.savefig(os.path.join(outdir,f"boxplot_{metric}_{save_name}.pdf"),bbox_inches='tight',pad_inches=0.02)
    plt.show()
    


def save_results_to_csv(method, mse_seen, mse_unseen, per_gene_mse_seen, per_gene_mse_unseen, correlations, seen_genes, unseen_genes):
    # Create a DataFrame for overall results
    overall_results = pd.DataFrame({
        'Metric': ['MSE', 'Average Per-Gene MSE'],
        'Seen Genes': [mse_seen, np.mean(per_gene_mse_seen)],
        'Unseen Genes': [mse_unseen, np.mean(per_gene_mse_unseen)]
    })

    # Create a DataFrame for per-gene results
    per_gene_results = pd.DataFrame({
        'Gene': seen_genes + unseen_genes,
        'Gene Type': ['Seen'] * len(seen_genes) + ['Unseen'] * len(unseen_genes),
        'MSE': np.concatenate([per_gene_mse_seen, per_gene_mse_unseen])
    })

    # Create a DataFrame for correlations of unseen genes
    correlation_df = pd.DataFrame(correlations, columns=['Gene', 'Correlation'])
    correlation_df['Gene Type'] = 'Unseen'

    # Combine per-gene results and correlations
    combined_results = pd.merge(per_gene_results, correlation_df, on=['Gene', 'Gene Type'], how='left')

    # Save to CSV
    with pd.ExcelWriter(f'{method}_results.xlsx') as writer:
        overall_results.to_excel(writer, sheet_name='Overall Results', index=False)
        combined_results.to_excel(writer, sheet_name='Per-Gene Results', index=False)

    print(f"Results saved to {method}_results.xlsx")


def envaluate_gene_expression(adata, module):
    y_true = adata.X.toarray()
    y_pred = adata.layers['predicted']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    per_gene_rmse = np.sqrt(np.mean((y_true.flatten() - y_pred.flatten()) ** 2, axis=0))
    pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    scc, _ = spearmanr(y_true.flatten(), y_pred.flatten()) 
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    print(f'Root Mean Squared Error on {module} Genes: {rmse}')
    print(f'Average Per-Gene RMSE on {module} Genes: {np.mean(per_gene_rmse)}')
    print(f'Pearson on {module} Genes: {pcc}')
    print(f'Spearman on {module} Genes: {scc}')
    print(f'R² on {module} Genes: {r2}')


def compute_correlations(y_unseen, y_unseen_pred, unseen_genes):
    """
    Computes Pearson correlation coefficients for unseen genes.
    """
    correlations = []
    for i, gene in enumerate(unseen_genes):
        original_expression = y_unseen[i, :]
        predicted_expression = y_unseen_pred[i, :]
        correlation, _ = pearsonr(original_expression, predicted_expression)
        correlations.append((gene, correlation))
    # Sort genes by correlation coefficient (descending order)
    correlations.sort(key=lambda x: x[1], reverse=True)
    return correlations


def visualize_results(per_gene_mse_seen, per_gene_mse_unseen, correlations, tsne_coords, y_unseen, y_unseen_pred, unseen_genes, top_n=5):
    """
    Visualizes the MSE distribution and top genes' expression.
    """
    # Plot MSE distribution
    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(per_gene_mse_seen, bins=50, alpha=0.5, label='Seen Data')
    plt.hist(per_gene_mse_unseen, bins=50, alpha=0.5, label='Unseen Data')
    plt.xlabel('Per-Gene MSE')
    plt.ylabel('Frequency')
    plt.title('Distribution of Per-Gene MSE for Seen and Unseen Data')
    plt.legend()
    plt.show()
    
    # Plot top genes
    top_genes = correlations[:top_n]
    fig, axes = plt.subplots(2, top_n, figsize=(4*top_n, 8), dpi=300)
    
    for i, (gene, correlation) in enumerate(top_genes):
        # Get the index of the gene
        gene_index = unseen_genes.index(gene)
        
        # Get the original and predicted expressions
        original_expression = y_unseen[gene_index, :]
        predicted_expression = y_unseen_pred[gene_index, :]
        
        # Plot original expression (top row)
        ax_original = axes[0, i]
        scatter_original = ax_original.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=original_expression, cmap='viridis', s=5)
        plt.colorbar(scatter_original, ax=ax_original, label='Expression')
        ax_original.set_xlabel('t-SNE 1')
        ax_original.set_ylabel('t-SNE 2')
        ax_original.set_title(f'{gene} (Original)\nCorrelation: {correlation:.4f}')
        
        # Plot predicted expression (bottom row)
        ax_predicted = axes[1, i]
        scatter_predicted = ax_predicted.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=predicted_expression, cmap='viridis', s=5)
        plt.colorbar(scatter_predicted, ax=ax_predicted, label='Expression')
        ax_predicted.set_xlabel('t-SNE 1')
        ax_predicted.set_ylabel('t-SNE 2')
        ax_predicted.set_title(f'{gene} (Predicted)')
    
        print(f"Correlation coefficient for {gene}: {correlation:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    # Plot correlation distribution
    all_correlations = [corr for _, corr in correlations]
    plt.figure(figsize=(6, 3), dpi=300)
    sns.histplot(all_correlations, kde=True)
    plt.xlabel('Pearson Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Distribution of Correlation Coefficients for Unseen Genes')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics for Correlation Coefficients:")
    print(f"Mean: {np.mean(all_correlations):.4f}")
    print(f"Median: {np.median(all_correlations):.4f}")
    print(f"Standard Deviation: {np.std(all_correlations):.4f}")
    print(f"Minimum: {np.min(all_correlations):.4f}")
    print(f"Maximum: {np.max(all_correlations):.4f}")
    
    

def plot_gene_spatial(
    gene_list, location, data,                     # data: {"GT": df, "CoxFormer": df, ...}
    save_dir=None, prefix="gene",
    layout="rows",                                 # "rows" or "pages"
    methods=None,                                  # default: list(data.keys())
    ncols=6, per_ax=1.0, dpi=200,
    cmap="RdBu_r", s=10, add_colorbar=True,
):
    if methods is None:
        methods = list(data.keys())
    gene_list = list(gene_list)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---- precompute expr + vmin/vmax per gene (shared across methods) ----
    payload = []
    for g in gene_list:
        expr = {m: (np.asarray(data[m][g], float) if (data.get(m) is not None and g in data[m].columns) else None)
                for m in methods}
        if layout == "single_row":
            gt = expr.get("GroundTruth", None)
            vmin_gt, vmax_gt = np.nanpercentile(gt, 1), np.nanpercentile(gt, 99)
            refs = [v for k, v in expr.items() if k != "GroundTruth"]
            pooled = np.concatenate([np.asarray(v).ravel() for v in refs])
            vmin_pred, vmax_pred = np.nanpercentile(pooled, 1), np.nanpercentile(pooled, 99)
            payload.append((g, expr, vmin_gt, vmax_gt, vmin_pred, vmax_pred))
        
        else:
            ref = next((v for v in expr.values() if v is not None and np.isfinite(v).any()), None)
            vmin, vmax = (np.nanpercentile(ref, 1), np.nanpercentile(ref, 99)) if ref is not None else (np.nan, np.nan)
            payload.append((g, expr, vmin, vmax))  
        

    def draw(ax, arr, title, vmin, vmax, show=True, show_title=False):
        if arr is None or (not np.isfinite(arr).any()):
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8)
            ax.axis("off")
            return
        plot_spatial_scatter(ax, location, arr, mode="continuous", cmap=cmap,
                             vmin=vmin, vmax=vmax, s=s, add_colorbar=add_colorbar,colorbar_size=4,
                             colorbar_width=0.1,colorbar_length=1, colorbar_linewidth=0.5,str_format=1,
                             colorbar_fraction=0.05, colorbar_pad=0, loc_inverse=False)
        if show_title:
            ax.set_title(title, fontsize=6)
        if show == True:
            for sp in ax.spines.values():
                sp.set_linewidth(0.5); sp.set_edgecolor("black")
        else:
            for sp in ax.spines.values():
                sp.set_visible(False)     

    # ---- layout: methods as rows (one figure) ----
    if layout == "rows":
        R, C = len(methods), len(gene_list)
        fig, axes = plt.subplots(R, C, figsize=(C*per_ax, R*per_ax), dpi=dpi, constrained_layout=True)
        axes = np.atleast_2d(axes)
        for j, (g, expr, vmin, vmax) in enumerate(payload):
            for i, m in enumerate(methods):
                draw(axes[i, j], expr[m], (g if i == 0 else ""), vmin, vmax)
        if save_dir:
            file_path = os.path.join(save_dir, f"{prefix}_rows.pdf")
            if not os.path.exists(file_path):
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
            else:
                print(f"{file_path} already exists, skip saving.")
        plt.show()

    if layout == "single_row":
        # one row, methods as columns, usually for a single gene
        g = gene_list[0]  # assume one gene; if multiple genes you can loop externally
        expr, vmin_gt, vmax_gt, vmin_pred, vmax_pred = payload[0][1], payload[0][2], payload[0][3], payload[0][4], payload[0][5]
        C = len(methods)
        fig, axes = plt.subplots(1, C, figsize=(C*per_ax, 1*per_ax), dpi=dpi, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for j, m in enumerate(methods):
            if m == "GroundTruth":
                vmin, vmax = vmin_gt, vmax_gt
            else:
                vmin, vmax = vmin_pred, vmax_pred
            draw(axes[j], expr[m], "", vmin, vmax, False)
        if save_dir:
            file_path = os.path.join(save_dir, f"{prefix}.pdf")
            if not os.path.exists(file_path):
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
            else:
                print(f"{file_path} already exists, skip saving.")
        plt.show()
        save_colorbar(vmin, vmax, cmap, save_dir, prefix)

    # ---- layout: one page per method (grid) ----
    if layout == "pages":
        N = len(gene_list)
        r = math.ceil(N / ncols)
        for m in methods:
            fig, axes = plt.subplots(r, ncols, figsize=(ncols*per_ax, r*per_ax), dpi=dpi, constrained_layout=True)
            axes = np.array(axes).ravel()
            for ax in axes[N:]:
                ax.axis("off")
            for i, (g, expr, vmin, vmax) in enumerate(payload):
                draw(axes[i], expr[m], g, vmin, vmax,show_title=True)
            if save_dir:
                file_path = os.path.join(save_dir, f"{prefix}_{m.replace(' ','_')}.pdf")
                if not os.path.exists(file_path):
                    plt.savefig(file_path, dpi=300, bbox_inches="tight")
                else:
                    print(f"{file_path} already exists, skip saving.")
            plt.show()

def save_colorbar(vmin, vmax, cmap, save_path, prefix, n_ticks=4,figsize=(1, 4),bold=False):
    # Create figure and axis with fixed size
    fig, ax = plt.subplots(figsize=figsize, dpi=300)  # Ensure consistent height
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Create the colorbar
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    # Set a fixed number of ticks on the colorbar, which will vary the tick spacing based on vmin and vmax
    tick_values = [vmin + (vmax - vmin) * i / (n_ticks - 1) for i in range(int(n_ticks))]
    cbar.set_ticks(tick_values)
    # Set tick parameters for clarity
    cbar.ax.tick_params(labelsize=20, labelcolor='black', width=1, length=8)
    cbar.ax.tick_params(axis='y', width=1)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.2f'))
    # Make the tick labels bold
    if bold:
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
    file_path = os.path.join(save_path,f"{prefix}_bar.pdf")
    if not os.path.exists(file_path):
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.02)
    else:
        print(f"{file_path} already exists, skip saving.")
    plt.close(fig)
    

def plot_spatial_scatter(ax, location, values, mode="continuous",cmap="cividis", color_map=None, order=None,add_colorbar=True, colorbar_size=10, colorbar_width=2, colorbar_length=4, colorbar_linewidth=2, str_format=2, colorbar_fraction=0.05,colorbar_pad=0.02, add_legend=False,
                         s=6, vmin=None, vmax=None, loc_inverse=True):
    
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    # --- coords normalize + swap ---
    x_raw = location["x"].to_numpy()
    y_raw = location["y"].to_numpy()
    x = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min() + 1e-9)
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min() + 1e-9)
    if loc_inverse:
        x, y = y, x
        
    fig = ax.get_figure()
    if mode == "continuous":
        v = values.to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
        sca = ax.scatter(
            x[m], y[m], c=v[m], s=s, cmap=cmap,
            vmin=vmin, vmax=vmax, linewidths=0, rasterized=True
        )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        if add_colorbar:
            vmin, vmax = sca.get_clim()
            ticks = np.linspace(vmin, vmax, 3)
            cbar = fig.colorbar(sca, ax=ax, ticks=ticks, fraction=colorbar_fraction, pad=colorbar_pad)
            cbar.ax.tick_params(labelsize=colorbar_size, width=colorbar_width, length=colorbar_length)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{str_format}f'))
            cbar.ax.tick_params(axis='y', pad=-0.5) 
            #for t in cbar.ax.get_yticklabels():
            #    t.set_fontweight('bold')
            cbar.outline.set_linewidth(colorbar_linewidth)

        return ax

    elif mode == "categorical":
        values = values.reindex(location.index)
        cat = values.astype("object")
        m = np.isfinite(x) & np.isfinite(y) & cat.notna().to_numpy()
        x, y, cat = x[m], y[m], cat[m]

        if color_map is None:
            raise ValueError("mode='categorical' requires color_map.")

        if order is None:
            order = list(color_map.keys())

        for name in order:
            mask = (cat == name).to_numpy()
            if mask.sum() == 0:
                continue
            ax.scatter(
                x[mask], y[mask], s=s,
                c=[color_map.get(name, "#999999")],
                linewidths=0, rasterized=True
            )

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        if add_legend:
            handles = [
                Line2D([0], [0], marker='o', linestyle='', markersize=6,
                       markerfacecolor=color_map.get(name, "#999999"),
                       markeredgecolor='none', label=name)
                for name in order
            ]
            ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=7)

        return ax

    else:
        raise ValueError("mode must be 'continuous' or 'categorical'.")
    
    