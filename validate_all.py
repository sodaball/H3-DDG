import os
from tqdm import tqdm
import pandas as pd
import torch
from utils import eval_skempi_three_modes
import time
import numpy as np
import argparse


def log_metrics_all(df_metrics, log_file, fold0_its=None, fold1_its=None, fold2_its=None):
    
    mode_names = ['all', 'single', 'multiple']
    metrics_str = (
        "\n"
        f"fold0: {fold0_its}, fold1: {fold1_its}, fold2: {fold2_its}"
        "\n"
        f"Mode {mode_names[0]}: "
        f"A-Pea {df_metrics[0][1]:.6f} A-Spe {df_metrics[0][2]:.6f} | "
        f"RMSE {df_metrics[0][3]:.6f} MAE {df_metrics[0][4]:.6f} AUROC {df_metrics[0][5]:.6f} AUPRC {df_metrics[0][6]:.6f} | "
        f"P-Pea {df_metrics[0][7]:.6f} P-Spe {df_metrics[0][8]:.6f}"
        "\n"
        f"Mode {mode_names[1]}: "
        f"A-Pea {df_metrics[1][1]:.6f} A-Spe {df_metrics[1][2]:.6f} | "
        f"RMSE {df_metrics[1][3]:.6f} MAE {df_metrics[1][4]:.6f} AUROC {df_metrics[1][5]:.6f} AUPRC {df_metrics[1][6]:.6f} | "
        f"P-Pea {df_metrics[1][7]:.6f} P-Spe {df_metrics[1][8]:.6f}"
        "\n"
        f"Mode {mode_names[2]}: "
        f"A-Pea {df_metrics[2][1]:.6f} A-Spe {df_metrics[2][2]:.6f} | "
        f"RMSE {df_metrics[2][3]:.6f} MAE {df_metrics[2][4]:.6f} AUROC {df_metrics[2][5]:.6f} AUPRC {df_metrics[2][6]:.6f} | "
        f"P-Pea {df_metrics[2][7]:.6f} P-Spe {df_metrics[2][8]:.6f}"
    )
    log_msg = f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [val]"
    log_msg += metrics_str
    print(f"\033[0;30;43m {log_msg}\033[0m")
    log_file.write(f"{log_msg}\n")
    log_file.flush()


def validate_all(save=False, save_dir=None, max_num_each_fold=5):

    fold0_its_list = []
    fold1_its_list = []
    fold2_its_list = []
    fold0_spearmans = []
    fold1_spearmans = []
    fold2_spearmans = []
    for fold in range(3):
        for csv_file in os.listdir(save_dir):
            if csv_file.endswith('_results.csv') and f'fold{fold}' in csv_file:
                its = csv_file.split('_')[1]
                if fold == 0:
                    fold0_its_list.append(its)
                    results = pd.read_csv(os.path.join(save_dir, csv_file))
                    results['num_muts'] = results['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                    df_metrics = eval_skempi_three_modes(results).to_numpy()
                    fold0_spearmans.append(df_metrics[0][1])
                elif fold == 1:
                    fold1_its_list.append(its)
                    results = pd.read_csv(os.path.join(save_dir, csv_file))
                    results['num_muts'] = results['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                    df_metrics = eval_skempi_three_modes(results).to_numpy()
                    fold1_spearmans.append(df_metrics[1][1])
                elif fold == 2:
                    fold2_its_list.append(its)
                    results = pd.read_csv(os.path.join(save_dir, csv_file))
                    results['num_muts'] = results['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                    df_metrics = eval_skempi_three_modes(results).to_numpy()
                    fold2_spearmans.append(df_metrics[2][1])

    # only take the top max_num_each_fold models for each fold
    fold0_its_list = sorted(fold0_its_list, key=lambda x: fold0_spearmans[fold0_its_list.index(x)], reverse=True)[:max_num_each_fold]
    fold1_its_list = sorted(fold1_its_list, key=lambda x: fold1_spearmans[fold1_its_list.index(x)], reverse=True)[:max_num_each_fold]
    fold2_its_list = sorted(fold2_its_list, key=lambda x: fold2_spearmans[fold2_its_list.index(x)], reverse=True)[:max_num_each_fold]
    
    if save:
        log_file = open(os.path.join(save_dir, 'validate_all.txt'), 'w')

    # permutation combination
    for fold0_its in tqdm(fold0_its_list, desc='validate_all', dynamic_ncols=True):
        for fold1_its in fold1_its_list:
            for fold2_its in fold2_its_list:
                results_path0 = os.path.join(save_dir, f'fold0_{fold0_its}_results.csv')
                results_path1 = os.path.join(save_dir, f'fold1_{fold1_its}_results.csv')
                results_path2 = os.path.join(save_dir, f'fold2_{fold2_its}_results.csv')
                if not os.path.exists(results_path0) or not os.path.exists(results_path1) or not os.path.exists(results_path2):
                    continue
                results0 = pd.read_csv(results_path0)
                results1 = pd.read_csv(results_path1)
                results2 = pd.read_csv(results_path2)
                # Convert num_muts column to tensor
                results0['num_muts'] = results0['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                results1['num_muts'] = results1['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                results2['num_muts'] = results2['num_muts'].apply(lambda x: eval(x, {'tensor': torch.tensor}))
                results = pd.concat([results0, results1, results2], ignore_index=True)
                df_metrics = eval_skempi_three_modes(results).to_numpy()
                log_metrics_all(df_metrics, log_file, fold0_its=fold0_its, fold1_its=fold1_its, fold2_its=fold2_its)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate all combinations of folds.")
    parser.add_argument('--top_k', type=int, default=5, required=False, help="Number of top models to consider from each fold.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory containing fold results.")
    parser.add_argument('--save', default=True, required=False, action='store_true', help="Whether to save the validation results.")
    args = parser.parse_args()

    validate_all(save=args.save, save_dir=args.save_dir, max_num_each_fold=args.top_k)