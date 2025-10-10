import os
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import wandb

from trainer import CrossValidation, recursive_to
from dataset import SkempiDatasetManager
from utils import set_seed, check_dir, eval_skempi_three_modes
from ddg_predictor import DDGPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_batch(model, batch, device, is_train=True, optimizer=None):
    """ Process a batch for training or validation. """
    batch = recursive_to(batch, device)
    if is_train:
        model.train()
        loss, output_dict, _ = model(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        optimizer.zero_grad()
    else:
        model.eval()
        with torch.no_grad():
            loss, output_dict, _ = model(batch)
    return loss, output_dict


def collect_results(model, dataloader, device):
    """ Collect prediction results from a dataloader. """
    results = []
    val_loss_list = []
    for batch in tqdm(dataloader, desc='validate', dynamic_ncols=True):
        loss, output_dict = process_batch(model, batch, device, is_train=False)
        val_loss_list.append(loss.item())
        for complex, num_muts, ddg_true, ddg_pred in zip(
            batch['complex'], batch['num_muts'], output_dict['ddG_true'], output_dict['ddG_pred']
        ):
            results.append({
                'complex': complex,
                'num_muts': num_muts,
                'ddG': ddg_true.item(),
                'ddG_pred': ddg_pred.item(),
            })
    return pd.DataFrame(results), np.mean(val_loss_list)


def train(max_iter, device='cpu', cv_mgr=None, dataloader=None, log_file=None, save_dir=None, args=None):
    """ Train the model across multiple cross-validation folds. """
    for fold in range(args.num_cvfolds):
        model, optimizer, _ = cv_mgr.get(fold)
        model.to(device)
        print(f"\033[0;30;43m{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] Fold {fold}\033[0m")

        for its in range(max_iter):
            batch = next(dataloader.get_train_loader(fold))
            if '3VR6_ABCDEF_GH' in batch['complex']: # remove 3VR6_ABCDEF_GH from train split, due to limit of GPU memory
                continue

            loss, _ = process_batch(model, batch, device, is_train=True, optimizer=optimizer)

            if its % 100 == 1:
                log_msg = f"{time.strftime('%Y-%m-%d %H-%M-%S')} | [train] iter {its} Fold {fold} | Loss {loss.item():.8f}"
                print(f"\033[0;30;46m{log_msg}\033[0m")
                log_file.write(f"{log_msg}\n")
                log_file.flush()

            if its == 1:    # check the program correctness
                online_validate(fold, save=True, save_threshold_fold0=args.save_threshold_fold0, save_threshold_fold1=args.save_threshold_fold1, save_threshold_fold2=args.save_threshold_fold2, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, its=its)
            
            if max_iter >= 50000:
                if its % args.val_freq == 1 and its >= 25000:
                    online_validate(fold, save=True, save_threshold_fold0=args.save_threshold_fold0, save_threshold_fold1=args.save_threshold_fold1, save_threshold_fold2=args.save_threshold_fold2, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, its=its)
                    if args.num_cvfolds == 1:
                        torch.save(cv_mgr.state_dict(), os.path.join(save_dir, 'checkpoint', f'ddg_model_{its}.ckpt'))
            else:
                if its % args.val_freq == 1:
                    online_validate(fold, save=True, save_threshold_fold0=args.save_threshold_fold0, save_threshold_fold1=args.save_threshold_fold1, save_threshold_fold2=args.save_threshold_fold2, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, its=its)
                    if args.num_cvfolds == 1:
                        torch.save(cv_mgr.state_dict(), os.path.join(save_dir, 'checkpoint', f'ddg_model_{its}.ckpt'))

        model.to('cpu')
        torch.cuda.empty_cache()


def validate(save=False, device='cpu', cv_mgr=None, dataloader=None, log_file=None, save_dir=None):
    """ Validate the model across all folds and compute metrics. """
    all_results = []
    for fold in range(args.num_cvfolds):
        model, _, _ = cv_mgr.get(fold)
        model.to(device)
        results, _ = collect_results(model, dataloader.get_val_loader(fold), device)
        all_results.append(results)
        model.to('cpu')

    results = pd.concat(all_results)
    results['method'] = 'BA-DDG'
    if save:
        results_path = os.path.join(save_dir, 'results.csv')
        results.to_csv(results_path, index=False)
        print(results_path)

    df_metrics = eval_skempi_three_modes(results).to_numpy()
    log_metrics(df_metrics, log_file, fold=None)

    return df_metrics
                

def online_validate(fold, save=False, save_threshold_fold0=0, save_threshold_fold1=0, save_threshold_fold2=0, device='cpu', cv_mgr=None, dataloader=None, log_file=None, its=None):
    """ Perform validation during training for a specific fold. """
    model, _, _ = cv_mgr.get(fold)
    results, val_loss = collect_results(model, dataloader.get_val_loader(fold), device)

    results['method'] = 'BA-DDG'
    df_metrics = eval_skempi_three_modes(results).to_numpy()
    log_metrics(df_metrics, log_file, fold=fold, val_loss=val_loss)
    if save:
        if fold == 0 and df_metrics[0][1] > save_threshold_fold0 or fold == 1 and df_metrics[1][1] > save_threshold_fold1 or fold == 2 and df_metrics[2][1] > save_threshold_fold2: # 超过阈值才保存预测结果
            results_path = os.path.join(save_dir, f'fold{fold}_its{its}_results.csv')
            results.to_csv(results_path, index=False)
            print(results_path)

    return df_metrics


def log_metrics(df_metrics, log_file, fold=None, val_loss=None):
    """ Log validation metrics to console, file, and optionally Weights & Biases. """
    mode_names = ['all', 'single', 'multiple']
    metrics_str = (
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
    if val_loss is not None:
        log_msg += f" loss {val_loss:.6f} | "
    log_msg += metrics_str

    print(f"\033[0;30;43m {log_msg}\033[0m")
    log_file.write(f"{log_msg}\n")
    log_file.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or infer with BA-DDG model.")
    parser.add_argument('--config_path', type=str, default="../config/inference_ba-cycle_skempi.json", help="Path to configuration JSON file.")
    parser.add_argument('--tag', type=str, default='', help="Tag for the experiment.")
    args = parser.parse_args()
    param_s = args.__dict__
    param = json.loads(open(args.config_path, 'r').read())
    param['tag'] = args.tag
    args = argparse.Namespace(**param)

    set_seed(args.seed)

    # Setup directories and logging
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    save_dir = os.path.join('./results/', timestamp + f"_{args.tag}")
    check_dir(os.path.join(save_dir, 'checkpoint'))
    log_file = open(os.path.join(save_dir, "train_log.txt"), 'a+')
    with open(os.path.join(save_dir, 'train_config.json'), 'w') as fout:
        json.dump(args.__dict__, fout, indent=2)

    print('Loading datasets...')
    dataloader = SkempiDatasetManager(config=args, num_cvfolds=args.num_cvfolds, num_workers=16)

    print('Building model...')
    cv_mgr = CrossValidation(config=args, num_cvfolds=args.num_cvfolds, model_factory=DDGPredictor).to('cpu')

    # Load pretrained weights if provided
    if 'ckpt_path' in args:
        cv_mgr.load_mpnn_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    if 'sft_ckpt_path' in args:
        cv_mgr.load_state_dict_inference(torch.load(args.sft_ckpt_path, map_location='cpu'))

    # Execute training or inference based on experiment type
    if args.ex_type == "inference":
        metrics = validate(save=True, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, save_dir=save_dir)
    elif args.ex_type == "train":
        train(max_iter=args.max_iter, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, save_dir=save_dir, args=args)
        torch.save(cv_mgr.state_dict(), os.path.join(save_dir, 'checkpoint', 'ddg_model.ckpt'))
        metrics = validate(save=True, device=device, cv_mgr=cv_mgr, dataloader=dataloader, log_file=log_file, save_dir=save_dir)
