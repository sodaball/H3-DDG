from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from protein_mpnn_utils import ProteinMPNN, _scores

class DDGPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mpnn = ProteinMPNN(ca_only=cfg.ca_only, num_letters=21, 
                node_features=cfg.hidden_dim, edge_features=cfg.hidden_dim, hidden_dim=cfg.hidden_dim, 
                num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers, augment_eps=cfg.backbone_noise, 
                k_neighbors=cfg.num_edges, num_tri_heads=cfg.num_tri_heads,
                use_hypergraph=cfg.use_hypergraph,
                max_num_hyperedges=cfg.max_num_hyperedges,
                num_mut_subgraph_nodes=cfg.num_mut_subgraph_nodes,
                hyper_ratio=cfg.hyper_ratio,
                num_edges_ratio=cfg.num_edges_ratio,
                edges_selection=cfg.edges_selection)
        self.boltzmann_scalar = nn.Parameter(torch.ones((1)))
        self.loss_weight_boltzmann = cfg.loss_weight_boltzmann
        
    def calc_thermodynamic_cycle(self, wt_scores, mut_scores, num_mut_chains, complex_indices, single_chain_indices, device):
        wt_complex_energy = wt_scores.index_select(0,complex_indices)
        wt_single_chain_energy = wt_scores.index_select(0,single_chain_indices) 
        wt_sum_single_chain_energy = torch.zeros_like(wt_complex_energy)
        wt_sum_single_chain_energy.scatter_add_(0, torch.arange(len(num_mut_chains), device=device).repeat_interleave(torch.tensor(num_mut_chains, device=device)), wt_single_chain_energy)
        wt_scores_cycle = (wt_complex_energy - wt_sum_single_chain_energy)

        mut_complex_energy = mut_scores.index_select(0,complex_indices)
        mut_single_chain_energy = mut_scores.index_select(0,single_chain_indices)
        mut_sum_single_chain_energy = torch.zeros_like(mut_complex_energy)
        mut_sum_single_chain_energy.scatter_add_(0, torch.arange(len(num_mut_chains), device=device).repeat_interleave(torch.tensor(num_mut_chains, device=device)), mut_single_chain_energy)
        mut_scores_cycle = (mut_complex_energy - mut_sum_single_chain_energy)

        return wt_scores_cycle, mut_scores_cycle
    
    def forward(self, batch):
        """
        Args of batch:
            X: (bs + num_mut_chains, L, 4, 3)
            aa: (bs + num_mut_chains, L)
            aa_mut: (bs + num_mut_chains, L)
            mask: (bs + num_mut_chains, L)
            chain_M: (bs + num_mut_chains, L)
            chain_encoding_all: (bs + num_mut_chains, L)
            residue_idx: (bs + num_mut_chains, L)
            complex: for example ['1C4Z_ABC_D', '1C4Z_ABC_D', '4L3E_ABC_DE', '4L3E_ABC_DE']
            ddG: (bs + num_mut_chains)
            id: (bs + num_mut_chains)
            num_muts: (bs + num_mut_chains)
            mutstr: for example ['IA163A', 'IA163A', 'YD26D,WE95L', 'YD26D,WE95L']
            num_mut_chains: for example [1, 1]/[1, 2]/[2, 1]/[2, 2]
        """

        log_probs = self.mpnn.deterministic_forward(batch['X'], batch['aa'], batch['mask'], batch['chain_M'], batch['residue_idx'], batch['chain_encoding_all'])

        wt_scores = _scores(batch['aa'], log_probs, batch['mask'])
        mut_scores = _scores(batch['aa_mut'], log_probs, batch['mask'])

        device = log_probs.device
        complex_indices = torch.concat((torch.tensor([0], device=device),torch.cumsum(torch.tensor(batch['num_mut_chains'], device=device), dim=0)), dim=-1)[:-1] + torch.arange(0, len(batch['num_mut_chains']), device=device)
        all_indices = torch.arange(0, log_probs.shape[0], device=device)
        single_chain_indices = all_indices[~torch.isin(all_indices, complex_indices)]

        wt_scores_cycle, mut_scores_cycle = self.calc_thermodynamic_cycle(wt_scores, mut_scores, batch['num_mut_chains'], complex_indices, single_chain_indices, device)
        
        mut_scores_cycle = mut_scores_cycle * self.boltzmann_scalar
        wt_scores_cycle = wt_scores_cycle * self.boltzmann_scalar
        ddg_pred = (mut_scores_cycle - wt_scores_cycle)

        loss = F.mse_loss(ddg_pred, batch['ddG'][complex_indices]) * self.loss_weight_boltzmann
        out_dict = {
            'ddG_pred': ddg_pred,
            'ddG_true': batch['ddG'][complex_indices],
        }
        return loss, out_dict, log_probs