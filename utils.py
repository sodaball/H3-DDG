import os
import yaml
import shutil
import random
import numpy as np
import pandas as pd

import torch
import torch.linalg
import torch.nn as nn

from tqdm.auto import tqdm
from easydict import EasyDict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LinearRegression


def get_edges_in_cluster(h_node_padded, Ca_padded, mask_node, num_edges_ratio, cluster_edge_mlp, edges_selection='random'):
    """
    Generates edge features and masks within padded clusters.
    Args:
        h_node_padded: (num_clusters, max_nodes, hidden_dim) Padded node features.
        Ca_padded: (num_clusters, max_nodes, 3) Padded C-alpha coordinates.
        mask_node: (num_clusters, max_nodes) Boolean mask for valid nodes.
        num_edges_ratio: Ratio to determine the number of edges per node.
        cluster_edge_mlp: MLP to generate edge features.
        edges_selection: Method for selecting edges ('random', 'distance_min', 'distance_max').
    Returns:
        h_edge_padded: (num_clusters, num_target_edges, hidden_dim) Padded edge features.
        mask_edge: (num_clusters, num_target_edges) Boolean mask for valid edges.
    """
    num_clusters, max_nodes, hidden_dim = h_node_padded.shape
    device = h_node_padded.device

    # Generate all possible pairwise edge features
    h_node_i = h_node_padded.unsqueeze(2).expand(-1, -1, max_nodes, -1)
    h_node_j = h_node_padded.unsqueeze(1).expand(-1, max_nodes, -1, -1)
    edge_features_all = torch.cat([h_node_i, h_node_j], dim=-1)
    h_edge_all = cluster_edge_mlp(edge_features_all)

    # Create mask for valid edges
    mask_node_i = mask_node.unsqueeze(2).expand(-1, -1, max_nodes)
    mask_node_j = mask_node.unsqueeze(1).expand(-1, max_nodes, -1)
    self_loop_mask = ~torch.eye(max_nodes, dtype=torch.bool, device=device).unsqueeze(0).expand(num_clusters, -1, -1)
    mask_edge_all = mask_node_i & mask_node_j & self_loop_mask

    # Flatten features and masks
    h_edge_all_flat = h_edge_all.view(num_clusters, max_nodes * max_nodes, hidden_dim)
    mask_edge_all_flat = mask_edge_all.view(num_clusters, max_nodes * max_nodes)

    # Determine the number of edges to select per cluster
    num_valid_nodes_per_cluster = mask_node.sum(dim=1)
    num_target_edges_per_cluster = torch.clamp((num_valid_nodes_per_cluster * num_edges_ratio).long(), min=1)

    num_possible_edges = num_valid_nodes_per_cluster * (num_valid_nodes_per_cluster - 1)
    num_target_edges_per_cluster = torch.min(num_target_edges_per_cluster, num_possible_edges)

    num_actual_valid_edges = mask_edge_all_flat.sum(dim=1)
    num_target_edges_per_cluster = torch.min(num_target_edges_per_cluster, num_actual_valid_edges)

    min_target_edges = num_target_edges_per_cluster.min().item() if num_clusters > 0 and (num_actual_valid_edges > 0).any() else 0

    if min_target_edges <= 0:
         return torch.zeros(num_clusters, 0, hidden_dim, device=device, dtype=h_node_padded.dtype), \
                torch.zeros(num_clusters, 0, dtype=torch.bool, device=device)

    num_target_edges = min_target_edges # Use the minimum target for padding consistency

    # Initialize output tensors
    h_edge_padded = torch.zeros(num_clusters, num_target_edges, hidden_dim, device=device, dtype=h_node_padded.dtype)
    mask_edge = torch.zeros(num_clusters, num_target_edges, dtype=torch.bool, device=device)


    # Select Edges based on criteria
    if edges_selection == 'random':
        for c in range(num_clusters):
            cluster_valid_indices = mask_edge_all_flat[c].nonzero(as_tuple=True)[0]
            num_valid_in_cluster = cluster_valid_indices.shape[0]

            if num_valid_in_cluster >= num_target_edges:
                perm = torch.randperm(num_valid_in_cluster, device=device)[:num_target_edges]
                sampled_indices_flat = cluster_valid_indices[perm]
                h_edge_padded[c] = h_edge_all_flat[c, sampled_indices_flat]
                mask_edge[c] = True
            elif num_valid_in_cluster > 0:
                sampled_indices_flat = cluster_valid_indices
                h_edge_padded[c, :num_valid_in_cluster] = h_edge_all_flat[c, sampled_indices_flat]
                mask_edge[c, :num_valid_in_cluster] = True

    elif edges_selection == 'distance_min':
        Ca_i = Ca_padded.unsqueeze(2).expand(-1, -1, max_nodes, -1)
        Ca_j = Ca_padded.unsqueeze(1).expand(-1, max_nodes, -1, -1)
        D_pair_sq = torch.sum((Ca_i - Ca_j)**2, dim=-1)

        D_pair_sq_masked = D_pair_sq.clone()
        D_pair_sq_masked[~mask_edge_all] = torch.finfo(D_pair_sq.dtype).max

        D_pair_flat = D_pair_sq_masked.view(num_clusters, max_nodes * max_nodes)

        sorted_dist, sorted_indices = torch.sort(D_pair_flat, dim=-1, descending=False)

        for c in range(num_clusters):
            valid_mask_flat_c = mask_edge_all_flat[c, sorted_indices[c]]
            valid_sorted_indices_c = sorted_indices[c][valid_mask_flat_c]

            num_valid_to_take = min(num_target_edges, valid_sorted_indices_c.shape[0])
            if num_valid_to_take > 0:
                topk_indices_c = valid_sorted_indices_c[:num_valid_to_take]
                h_edge_padded[c, :num_valid_to_take] = h_edge_all_flat[c, topk_indices_c]
                mask_edge[c, :num_valid_to_take] = True

    elif edges_selection == 'distance_max':
        Ca_i = Ca_padded.unsqueeze(2).expand(-1, -1, max_nodes, -1)
        Ca_j = Ca_padded.unsqueeze(1).expand(-1, max_nodes, -1, -1)
        D_pair_sq = torch.sum((Ca_i - Ca_j)**2, dim=-1)

        D_pair_sq_masked = D_pair_sq.clone()
        D_pair_sq_masked[~mask_edge_all] = -torch.finfo(D_pair_sq.dtype).max

        D_pair_flat = D_pair_sq_masked.view(num_clusters, max_nodes * max_nodes)

        sorted_dist, sorted_indices = torch.sort(D_pair_flat, dim=-1, descending=True)

        for c in range(num_clusters):
            valid_mask_flat_c = mask_edge_all_flat[c, sorted_indices[c]]
            valid_sorted_indices_c = sorted_indices[c][valid_mask_flat_c]

            num_valid_to_take = min(num_target_edges, valid_sorted_indices_c.shape[0])
            if num_valid_to_take > 0:
                topk_indices_c = valid_sorted_indices_c[:num_valid_to_take]
                h_edge_padded[c, :num_valid_to_take] = h_edge_all_flat[c, topk_indices_c]
                mask_edge[c, :num_valid_to_take] = True

    elif edges_selection == 'dynamic':
        # compute pairwise distances
        Ca_i = Ca_padded.unsqueeze(2).expand(-1, -1, max_nodes, -1)
        Ca_j = Ca_padded.unsqueeze(1).expand(-1, max_nodes, -1, -1)
        D_pair_sq = torch.sum((Ca_i - Ca_j)**2, dim=-1)
        distance_threshold_sq = 8.0 ** 2

        # prepare short edge mask
        short_edge_mask = D_pair_sq < distance_threshold_sq
        valid_short_edge_mask = mask_edge_all & short_edge_mask
        valid_short_edge_mask_flat = valid_short_edge_mask.view(num_clusters, max_nodes * max_nodes)

        # prepare distances for alternative (distance_min)
        D_pair_sq_masked_for_min = D_pair_sq.clone()
        D_pair_sq_masked_for_min[~mask_edge_all] = torch.finfo(D_pair_sq.dtype).max
        D_pair_flat_for_min = D_pair_sq_masked_for_min.view(num_clusters, max_nodes * max_nodes)

        for c in range(num_clusters):
            valid_short_indices_flat_c = valid_short_edge_mask_flat[c].nonzero(as_tuple=True)[0]
            num_valid_short_in_cluster = valid_short_indices_flat_c.shape[0]

            if num_valid_short_in_cluster >= num_target_edges:
                perm = torch.randperm(num_valid_short_in_cluster, device=device)[:num_target_edges]
                sampled_indices_flat = valid_short_indices_flat_c[perm]
                h_edge_padded[c] = h_edge_all_flat[c, sampled_indices_flat]
                mask_edge[c] = True
            else:
                cluster_valid_indices = mask_edge_all_flat[c].nonzero(as_tuple=True)[0]
                num_valid_in_cluster = cluster_valid_indices.shape[0]

                num_to_take = min(num_target_edges, num_valid_in_cluster)

                if num_to_take > 0:
                    valid_distances_c = D_pair_flat_for_min[c, cluster_valid_indices]
                    _, topk_indices_in_valid = torch.topk(valid_distances_c, num_to_take, largest=False, sorted=False)
                    topk_indices_flat = cluster_valid_indices[topk_indices_in_valid]

                    h_edge_padded[c, :num_to_take] = h_edge_all_flat[c, topk_indices_flat]
                    mask_edge[c, :num_to_take] = True

    else:
         raise ValueError(f"Unknown edges_selection method: {edges_selection}. Please choose from: random, distance_min, distance_max, dynamic.")

    return h_edge_padded, mask_edge


class ScaleAwareFusion(nn.Module):
    def __init__(self, dim, alpha_init=0.01, alpha_max=0.1, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.dropout = nn.Dropout(dropout)
        self.alpha_max = alpha_max
        self.alpha_init = alpha_init

    def forward(self, h_V, h_V_hyper_to_full):
        h_V_hyper_to_full = self.norm(h_V_hyper_to_full)
        alpha = self.alpha

        h_V_scaled = self.dropout(alpha * h_V_hyper_to_full)
        h_V_fused = h_V + h_V_scaled
        
        return h_V_fused


def hyper_pooling(h_V, Ca, mask, chain_M, hyper_ratio=4):
    """
    Hyper-pooling with FPS sampling and cluster mean pooling. L_hyper is dynamic.
    Args:
        h_V: (B, L, D) node features
        Ca: (B, L, 3) Cα coordinates
        mask: (B, L) residue mask (1 for valid)
        chain_M: (B, L) chain mask (1 for mutated/target)
        hyper_ratio: Ratio to determine L_hyper (L_hyper = valid_L // hyper_ratio)
    Returns:
        h_V_hyper_padded: (B, max_k, D) pooled features, padded
        mask_hyper_padded: (B, max_k) hyper-node mask, padded (boolean)
        full_to_hyper: (B, L) mapping from original to hyper nodes
        mutated_cluster_ids_batch: List[List[int]] containing unique mutated cluster IDs for each sample
    """
    B, L, D = h_V.shape
    device = h_V.device

    valid_L_list = mask.sum(dim=1).long()
    k_list = torch.where(valid_L_list > 0, torch.clamp(valid_L_list // hyper_ratio, min=1), 0)
    max_k = k_list.max().item() if B > 0 else 0

    h_V_hyper_padded = torch.zeros(B, max_k, D, device=device, dtype=h_V.dtype)
    mask_hyper_padded = torch.zeros(B, max_k, dtype=torch.bool, device=device)
    full_to_hyper = torch.zeros(B, L, dtype=torch.long, device=device) # (B, L)

    mutated_cluster_ids_batch = []

    for b in range(B):
        valid_mask_b = mask[b]  # (L,)
        valid_indices = valid_mask_b.nonzero(as_tuple=True)[0]
        valid_L = valid_indices.shape[0]
        k = k_list[b].item()

        if valid_L == 0 or k == 0:
            mutated_cluster_ids_batch.append([])
            continue

        valid_Ca = Ca[b, valid_indices]
        valid_chain_M = chain_M[b, valid_indices]
        mutated_indices_in_valid = torch.nonzero(valid_chain_M, as_tuple=True)[0]

        actual_k = k
        # use FPS to sample actual_k points from valid_Ca
        fps_idx_in_valid = farthest_point_sample(valid_Ca, actual_k, mutated_indices_in_valid)  # (actual_k,)

        actual_k = fps_idx_in_valid.shape[0]
        if actual_k == 0:
             mutated_cluster_ids_batch.append([])
             continue

        fps_Ca = valid_Ca[fps_idx_in_valid]  # (actual_k, 3)

        # assign cluster id to all valid nodes
        distances = torch.cdist(valid_Ca, fps_Ca)  # (valid_L, actual_k)
        cluster_ids_in_valid = torch.argmin(distances, dim=1)  # (valid_L,)
        full_to_hyper[b, valid_indices] = cluster_ids_in_valid

        # store unique mutated cluster ids for this sample
        if mutated_indices_in_valid.numel() > 0:
             mutated_clusters = cluster_ids_in_valid[mutated_indices_in_valid].unique().tolist()
             mutated_cluster_ids_batch.append(mutated_clusters)
        else:
             mutated_cluster_ids_batch.append([])

        # mean pool over valid nodes in each cluster
        h_V_b_valid = h_V[b, valid_indices] # (valid_L, D)
        for j in range(actual_k):
            cluster_node_mask_in_valid = (cluster_ids_in_valid == j)
            if cluster_node_mask_in_valid.any():
                h_V_hyper_padded[b, j] = h_V_b_valid[cluster_node_mask_in_valid].mean(dim=0)
        
        mask_hyper_padded[b, :actual_k] = True

    return h_V_hyper_padded, mask_hyper_padded, full_to_hyper, mutated_cluster_ids_batch


def hyper_to_full_graph(h_V_hyper, mask_hyper, full_to_hyper):
    """
    Propagate hyper-node features back to original graph.
    Handles padded h_V_hyper and mask_hyper.
    Args:
        h_V_hyper: (B, max_k, D) hyper-node features (PADDED)
        mask_hyper: (B, max_k) hyper-node mask (PADDED, boolean)
        full_to_hyper: (B, L) mapping from original to hyper nodes (values 0 to k-1)
    Returns:
        h_V_full: (B, L, D) propagated features
    """
    B, L = full_to_hyper.shape
    D = h_V_hyper.shape[-1]

    max_k = h_V_hyper.shape[1]
    clamped_full_to_hyper = torch.clamp(full_to_hyper, max=max_k-1)
    h_V_full = h_V_hyper.gather(1, clamped_full_to_hyper.unsqueeze(-1).expand(-1, -1, D))
    valid_hyper_mask = mask_hyper.gather(1, clamped_full_to_hyper)  # (B, L)

    return h_V_full * valid_hyper_mask.unsqueeze(-1).to(h_V_full.dtype)


def farthest_point_sample(xyz, n_sample, mutated_indices):
    """
    Uses FPS to sample n_sample points from xyz.
    Prioritizes mutated_indices if provided and n_sample allows.
    Args:
        xyz: (N, 3) float32 array of points
        n_sample: int, number of points to sample
        mutated_indices: (M,) long tensor of indices to prioritize, can be empty.
    Returns:
        centroids: (num_actually_sampled,) long tensor of indices of sampled points
                   num_actually_sampled = min(n_sample, N)
    """
    N, C = xyz.shape
    device = xyz.device

    num_to_sample = min(n_sample, N)
    if num_to_sample <= 0:
        return torch.tensor([], dtype=torch.long, device=device)
    centroids = torch.zeros(num_to_sample, dtype=torch.long, device=device)
    distance = torch.ones(N, device=device) * 1e10

    num_mutated = mutated_indices.shape[0]
    start_idx = 0

    if num_mutated > 0:
        num_priority = min(num_mutated, num_to_sample)
        priority_indices = torch.unique(mutated_indices[mutated_indices < N])[:num_priority]
        num_priority = priority_indices.shape[0]

        if num_priority > 0:
            centroids[:num_priority] = priority_indices
            xyz_priority = xyz[priority_indices] # (num_priority, 3)
            dist_to_priority = torch.cdist(xyz, xyz_priority) # (N, num_priority)
            distance = torch.min(dist_to_priority, dim=1)[0]
            distance[priority_indices] = -1e10  # avoid being selected again
            start_idx = num_priority

    # continue FPS to select remaining points
    if start_idx < num_to_sample:
        # if start_idx is 0, select a random starting point
        # otherwise, select the point farthest from the selected priority points
        if start_idx == 0:
             farthest = torch.randint(0, N, (1,), dtype=torch.long, device=device)[0]
        else:
             farthest = torch.argmax(distance)

        centroids[start_idx] = farthest
        xyz_farthest = xyz[farthest].view(1, C)
        dist = torch.cdist(xyz, xyz_farthest).view(-1) # (N,)
        distance = torch.min(distance, dist)
        distance[farthest] = -1e10
        start_idx += 1
    
    for i in range(start_idx, num_to_sample):
        # find the point farthest from the current selected points
        farthest = torch.argmax(distance)
        centroids[i] = farthest
        xyz_farthest = xyz[farthest].view(1, C)
        dist = torch.cdist(xyz, xyz_farthest).view(-1) # (N,)
        distance = torch.min(distance, dist)
        distance[farthest] = -1e10

    return centroids


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def check_dir(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def save_code(save_dir):
    save_code_dir = os.path.join(save_dir, 'codes/')
    check_dir(save_code_dir + 'common_utils/modules/')
    check_dir(save_code_dir + 'common_utils/protein/')
    check_dir(save_code_dir + 'common_utils/transforms/')

    for file in os.listdir("../code"):
        if '.py' in file:
            shutil.copyfile('../code/' + file, save_code_dir + file)
    for file in os.listdir("../code/common_utils/modules/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/modules/' + file, save_code_dir + 'common_utils/modules/' + file)
    for file in os.listdir("../code/common_utils/protein/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/protein/' + file, save_code_dir + 'common_utils/protein/' + file)
    for file in os.listdir("../code/common_utils/transforms/"):
        if '.py' in file:
            shutil.copyfile('../code/common_utils/transforms/' + file, save_code_dir + 'common_utils/transforms/' + file)


class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream):
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.load(f, Loader))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    return config, config_name


def per_complex_corr(df, pred_attr='ddG_pred', limit=10):
    corr_table = []

    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')

        if len(df_cplx) < limit: 
            continue

        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', pred_attr]].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', pred_attr]].corr('spearman').iloc[0,1],
        })

    corr_table = pd.DataFrame(corr_table)
    avg = corr_table[['pearson', 'spearman']].mean()

    return avg['pearson'] , avg['spearman']


def overall_correlations(df):
    pearson = df[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1]
    spearman = df[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1]
    return {
        'overall_pearson': pearson, 
        'overall_spearman': spearman,
    }


def percomplex_correlations(df, return_details=False):
    corr_table = []
    for cplx in df['complex'].unique():
        df_cplx = df.query(f'complex == "{cplx}"')
        if len(df_cplx) < 10: 
            continue
        corr_table.append({
            'complex': cplx,
            'pearson': df_cplx[['ddG', 'ddG_pred']].corr('pearson').iloc[0,1],
            'spearman': df_cplx[['ddG', 'ddG_pred']].corr('spearman').iloc[0,1],
        })
    corr_table = pd.DataFrame(corr_table)
    average = corr_table[['pearson', 'spearman']].mean()
    out = {
        'percomplex_pearson': average['pearson'],
        'percomplex_spearman': average['spearman'],
    }
    if return_details:
        return out, corr_table
    else:
        return out


def overall_auroc(df):
    score = roc_auc_score(
        (df['ddG'] > 0).to_numpy(),
        df['ddG_pred'].to_numpy()
    )
    return {
        'auroc': score,
    }


def overall_auprc(df):
    score = average_precision_score(
        (df['ddG'] > 0).to_numpy(),
        df['ddG_pred'].to_numpy()
    )
    return {
        'auprc': score,
    }


def overall_rmse_mae(df):
    true = df['ddG'].to_numpy()
    pred = df['ddG_pred'].to_numpy()[:, None]
    reg = LinearRegression().fit(pred, true)
    pred_corrected = reg.predict(pred)
    rmse = np.sqrt( ((true - pred_corrected) ** 2).mean() )
    mae = np.abs(true - pred_corrected).mean()
    return {
        'rmse': rmse,
        'mae': mae,
    }


def analyze_all_results(df):
    methods = df['method'].unique()
    funcs = [
        overall_correlations,
        overall_rmse_mae,
        overall_auroc,
        overall_auprc,
        percomplex_correlations,
    ]
    analysis = []
    for method in methods:
        df_this = df[df['method'] == method]
        result = {
            'method': method,
        }
        for f in funcs:
            result.update(f(df_this))
        analysis.append(result)
    analysis = pd.DataFrame(analysis)
    return analysis


def analyze_all_percomplex_correlations(df):
    methods = df['method'].unique()
    df_corr = []
    for method in methods:
        df_this = df[df['method'] == method]
        _, df_corr_this = percomplex_correlations(df_this, return_details=True)
        df_corr_this['method'] = method
        df_corr.append(df_corr_this)
    df_corr = pd.concat(df_corr).reset_index()
    return df_corr


def eval_skempi(df_items, mode, ddg_cutoff=None):
    assert mode in ('all', 'single', 'multiple')
    if mode == 'single':
        df_items = df_items.query('num_muts == 1')
    elif mode == 'multiple':
        df_items = df_items.query('num_muts > 1')

    if ddg_cutoff is not None:
        df_items = df_items.query(f"ddG >= {-ddg_cutoff} and ddG <= {ddg_cutoff}")

    df_metrics = analyze_all_results(df_items)
    df_corr = analyze_all_percomplex_correlations(df_items)
    df_metrics['mode'] = mode
    return df_metrics


def eval_skempi_three_modes(results, ddg_cutoff=None):
    df_all = eval_skempi(results, mode='all', ddg_cutoff=ddg_cutoff)
    df_single = eval_skempi(results, mode='single', ddg_cutoff=ddg_cutoff)
    df_multiple = eval_skempi(results, mode='multiple', ddg_cutoff=ddg_cutoff)
    df_metrics = pd.concat([df_all, df_single, df_multiple], axis=0)
    df_metrics.reset_index(inplace=True, drop=True)
    return df_metrics
