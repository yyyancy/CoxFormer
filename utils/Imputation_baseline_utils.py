# ======== Basic utilities ========
import os
import sys
import copy
import warnings
from os.path import join

# ======== Scientific computing ========
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

# ======== Bioinformatics ========
import scanpy as sc

# ======== Warning settings ========
warnings.filterwarnings('ignore')

def SpaGE_impute():
    print ('We run SpaGE for this data\n')
    sys.path.append(os.path.abspath("../Extenrnal/SpaGE-master/"))
    from SpaGE.main import SpaGE
    global RNA_data, Spatial_data, train_gene, predict_gene
    RNA_data = RNA_data.loc[:,(RNA_data.sum(axis=0) != 0)]
    RNA_data = RNA_data.loc[:,(RNA_data.var(axis=0) != 0)]
    train = np.array(list(set(train_gene) & set(RNA_data.columns)))
    predict = np.array(list(set(predict_gene) & set(RNA_data.columns)))
    print(f"train:{len(train)},predict:{len(predict)}")
    pv = len(train)/2
    Spatial = Spatial_data[train]
    Img_Genes = SpaGE(Spatial, RNA_data, n_pv = int(pv), genes_to_predict = predict)
    result = Img_Genes[predict]
    return result

def gimVI_impute():
    print ('We run gimVI for this data\n')
    import scvi
    scvi.settings.dl_pin_memory = False
    scvi.settings.dl_num_workers = 0
    import scanpy as sc
    target_dir = os.path.abspath("../Extenrnal/scvi/model")
    sys.path.insert(0, target_dir)
    from gimvi import GIMVI
    global RNA_adata, Spatial_adata, train_gene, predict_gene
    test_list = np.array(predict_gene)
    train_list = np.array(train_gene)
    Genes  = list(Spatial_adata.var_names)
    rand_gene_idx = [Genes.index(x) for x in test_list]
    n_genes = len(Genes)
    rand_train_gene_idx = sorted(set(range(n_genes)) - set(rand_gene_idx))
    rand_train_genes = np.array(Genes)[rand_train_gene_idx]
    rand_genes = np.array(Genes)[rand_gene_idx]
    spatial_data_partial = Spatial_adata[:, rand_train_genes]
    sc.pp.filter_cells(spatial_data_partial, min_counts= 0)
    seq_data = copy.deepcopy(RNA_adata)
    seq_data = seq_data[:, Genes]
    sc.pp.filter_cells(seq_data, min_counts = 0)
    scvi.data.setup_anndata(spatial_data_partial)
    scvi.data.setup_anndata(seq_data)
    model = GIMVI(seq_data, spatial_data_partial)
    model.train(200)
    _, imputation = model.get_imputed_values(normalized = False)
    imputed = imputation[:, rand_gene_idx]
    result = pd.DataFrame(imputed, columns = rand_genes)
    return result
                       
def novoSpaRc_impute():
    print ('We run novoSpaRc for this data\n')
    import novosparc as nc
    global RNA_data, Spatial_data, locations, train_gene, predict_gene
    test_list = np.array(predict_gene)
    train_list = np.array(train_gene)
    gene_names = np.array(RNA_data.columns.values)
    dge = RNA_data.values
    num_cells = dge.shape[0]
    print ('number of cells and genes in the matrix:', dge.shape)
    hvg = np.argsort(np.divide(np.var(dge, axis = 0),np.mean(dge, axis = 0) + 0.0001))
    dge_hvg = dge[:,hvg[-2000:]]
    num_locations = locations.shape[0]
    p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
    cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg,locations,num_neighbors_source = 5,num_neighbors_target = 5)
    spatial_genes = np.array(Spatial_data.columns)
    insitu_genes = np.intersect1d(train_list, spatial_genes)
    markers_in_sc = np.where(np.isin(gene_names, insitu_genes))[0]
    if len(markers_in_sc) == 0:
        raise ValueError("No overlapping marker genes between RNA_data and Spatial_data.")
    matched_genes = gene_names[markers_in_sc]
    dge_sub = dge[:, markers_in_sc]                                # cells × G
    insitu_matrix = Spatial_data.loc[:, matched_genes].to_numpy()   # spots × G
    print(f"markers_in_sc:{markers_in_sc.shape}, dge:{dge[:, markers_in_sc].shape} ,insitu_matrix:{insitu_matrix.shape}")
    cost_marker_genes = cdist(dge[:, markers_in_sc]/np.amax(dge[:, markers_in_sc]),insitu_matrix/np.amax(insitu_matrix))
    alpha_linear = 0.5
    gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,alpha_linear, p_expression, p_location,'square_loss', epsilon=5e-3, verbose=True)
    sdge = np.dot(dge.T, gw)
    imputed = pd.DataFrame(sdge,index = RNA_data.columns)
    result = imputed.loc[list(set(test_list) & set(RNA_data.columns))]
    result = result.T
    return result
                       
def SpaOTsc_impute():
    print ('We run SpaOTsc for this data\n')
    sys.path.append(os.path.abspath("../Extenrnal/SpaOTsc/"))
    from spaotsc import SpaOTsc
    global RNA_data, Spatial_data, locations, train_gene, predict_gene
    test_list = np.array(predict_gene)
    train_list = np.array(train_gene)
    df_sc = RNA_data
    df_IS = Spatial_data
    pts = locations
    is_dmat = distance_matrix(pts, pts)
    df_is = df_IS.loc[:,train_list]
    gene_is = df_is.columns.tolist()
    gene_sc = df_sc.columns.tolist()
    gene_overloap = list(set(gene_is).intersection(gene_sc))
    a = df_is[gene_overloap]
    b = df_sc[gene_overloap]
    rho, pval = stats.spearmanr(a, b,axis=1)
    rho[np.isnan(rho)]=0
    mcc=rho[-(len(df_sc)):,0:len(df_is)]
    C = np.exp(1 - mcc)
    issc = SpaOTsc.spatial_sc(sc_data = df_sc, is_data = df_is, is_dmat = is_dmat)
    issc.transport_plan(C**2, alpha = 0, rho = 1.0, epsilon = 0.1, cor_matrix = mcc, scaling = False)
    gamma = issc.gamma_mapping
    for j in range(gamma.shape[1]):
        gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
    X_pred = np.matmul(gamma.T, np.array(issc.sc_data.values))
    result = pd.DataFrame(data = X_pred, columns = issc.sc_data.columns.values)
    test_genes = test_list
    result = result.loc[:, test_genes]
    return result

def stPlus_impute():
    target_dir = os.path.abspath("../Extenrnal/stPlus/")
    sys.path.insert(0, target_dir)
    from model import stPlus
    global RNA_data, Spatial_data, outdir, train_gene, predict_gene
    train_list = np.array(train_gene)
    test_list = np.array(predict_gene)
    save_path_prefix = join(outdir, 'process_file/stPlus-demo')
    if not os.path.exists(join(outdir, "process_file")):
        os.mkdir(join(outdir, "process_file"))
    stPlus_res = stPlus(Spatial_data[train_list], RNA_data, test_list, save_path_prefix)
    return stPlus_res
                       
def Tangram_impute(annotate = None, modes = 'clusters', density = 'rna_count_based'):
    sys.path.append(os.path.abspath("../Extenrnal/"))
    import torch
    import tangram as tg
    print ('We run Tangram for this data')
    global RNA_adata, Spatial_adata, locations, train_gene, predict_gene
    test_list = predict_gene
    test_list = [x.lower() for x in test_list]
    train_list = train_gene
    spatial_data_partial = Spatial_adata[:, train_list]
    train_list = np.array(train_list)
    if annotate == None:
        RNA_adata_label = RNA_adata.copy()
        sc.pp.highly_variable_genes(RNA_adata_label)
        RNA_adata_label = RNA_adata_label[:, RNA_adata_label.var.highly_variable]
        sc.pp.scale(RNA_adata_label, max_value=10)
        sc.tl.pca(RNA_adata_label)
        sc.pp.neighbors(RNA_adata_label)
        sc.tl.leiden(RNA_adata_label, resolution = 0.5)
        RNA_adata.obs['leiden']  = RNA_adata_label.obs.leiden
    else:
        global CellTypeAnnotate
        RNA_adata.obs['leiden']  = CellTypeAnnotate
    tg.pp_adatas(RNA_adata, spatial_data_partial, genes=train_list)
    device = torch.device('cuda:0')
    if modes == 'clusters':
        ad_map = tg.map_cells_to_space(RNA_adata, spatial_data_partial, device = device, mode = modes, cluster_label = 'leiden', density_prior = density)
        ad_ge = tg.project_genes(ad_map, RNA_adata, cluster_label = 'leiden')
    else:
        ad_map = tg.map_cells_to_space(RNA_adata, spatial_data_partial, device = device)
        ad_ge = tg.project_genes(ad_map, RNA_adata)
    test_list = list(set(ad_ge.var_names) & set(test_list))
    test_list = np.array(test_list)
    pre_gene = pd.DataFrame(ad_ge[:,test_list].X, index=ad_ge[:,test_list].obs_names, columns=ad_ge[:,test_list].var_names.str.upper())
    return pre_gene