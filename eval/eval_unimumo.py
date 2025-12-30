import sys
import os
import torch
import numpy as np
from datetime import datetime
from collections import OrderedDict

# 引用現有的 utils
from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader

# 引用還原 3D 骨架所需的函式
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

def calculate_diversity(features, diversity_times):
    """
    計算 Diversity (即 Dist_k 或 Dist_g)
    features: (Num_Samples, Feature_Dim)
    diversity_times: 重複計算次數
    """
    num_samples = features.shape[0]
    if num_samples < diversity_times:
        #print(f"Warning: Sample size ({num_samples}) is smaller than diversity_times ({diversity_times}). Adjusting diversity_times to {num_samples}.")
        diversity_times = num_samples

    first_indices = np.random.randint(0, num_samples, diversity_times)
    second_indices = np.random.randint(0, num_samples, diversity_times)
    
    dist_sum = 0
    for i in range(diversity_times):
        # 計算兩兩樣本間的歐式距離
        dist = np.linalg.norm(features[first_indices[i]] - features[second_indices[i]])
        dist_sum += dist
    
    return dist_sum / diversity_times

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

def extract_features(joints):
    """
    從 3D 關節提取 Geometric 和 Kinetic 特徵
    joints: (Batch, Frames, Joints, 3)
    """
    # 1. Geometric Features: 直接使用關節位置
    # 形狀: (Batch, Frames * Joints * 3)
    # 為了避免長度不一，通常取固定的 slice 或是 flatten 整個序列 (假設長度一致)
    batch_size, frames, num_joints, dims = joints.shape
    geometric_feats = joints.reshape(batch_size, -1)

    # 2. Kinetic Features: 計算速度 (Velocity)
    # Velocity = Position[t+1] - Position[t]
    # 我們補一個 0 在最後以維持維度一致，或者只取 T-1
    velocity = np.diff(joints, axis=1) # (Batch, Frames-1, Joints, 3)
    # Pad last frame with zero velocity to keep shape
    velocity = np.pad(velocity, ((0,0), (0,1), (0,0), (0,0)), mode='constant')
    kinetic_feats = velocity.reshape(batch_size, -1)

    return geometric_feats, kinetic_feats



def evaluate_ground_truth(loader, args):
    print("Evaluating Ground Truth...")
    all_geometric_feats = []
    all_kinetic_feats = []
    
    dataset = loader.dataset
    
    # 預先處理 mean/std 形狀
    mean = dataset.t2m_dataset.mean
    std = dataset.t2m_dataset.std
    if mean.ndim > 1: mean = mean.squeeze()
    if std.ndim > 1: std = std.squeeze()
    mean = torch.from_numpy(mean).to(dist_util.dev())
    std = torch.from_numpy(std).to(dist_util.dev())

    # 1. 提取所有 GT 特徵 (只需做一次)
    for batch in loader:
        if isinstance(batch, tuple) and len(batch) == 2:
            motions, _ = batch
        elif isinstance(batch, list) or (isinstance(batch, tuple) and len(batch) >= 7):
            if len(batch) == 8:
                _, _, _, _, motions, m_lens, _, _ = batch
            else:
                _, _, _, _, motions, m_lens, _ = batch
        else:
            continue

        motions = motions.to(dist_util.dev())
        
        if motions.ndim == 4:
            motions = motions.squeeze(2) 
        motions = motions.permute(0, 2, 1) 
        
        motions = motions * std + mean
        motions = motions.float()
        
        n_joints = 22 if motions.shape[-1] == 263 else 21
        sample_xyz = recover_from_ric(motions, n_joints)
        sample_xyz = sample_xyz.cpu().numpy()
        
        g_feat, k_feat = extract_features(sample_xyz)
        all_geometric_feats.append(g_feat)
        all_kinetic_feats.append(k_feat)

    if len(all_geometric_feats) == 0:
        print("Error: No valid batches found in loader!")
        return

    all_geometric_feats = np.concatenate(all_geometric_feats, axis=0)
    all_kinetic_feats = np.concatenate(all_kinetic_feats, axis=0)
    
    # 2. 多次計算 Diversity 取平均
    dist_g_list = []
    dist_k_list = []
    
    # 判斷採樣數量
    div_times = args.diversity_times
    if all_geometric_feats.shape[0] < div_times:
        div_times = all_geometric_feats.shape[0]
        
    print(f"Calculating GT Metrics over {args.replication_times} replications (sample size: {div_times})...")

    for i in range(args.replication_times):
        dist_g = calculate_diversity(all_geometric_feats, div_times)
        dist_k = calculate_diversity(all_kinetic_feats, div_times)
        
        dist_g_list.append(dist_g)
        dist_k_list.append(dist_k)
        
        # 選擇性：印出每次結果
        # print(f"  GT Rep {i+1}: Dist_g={dist_g:.4f}, Dist_k={dist_k:.4f}")

    # 3. 統計結果
    mean_g, conf_g = get_metric_statistics(np.array(dist_g_list), args.replication_times)
    mean_k, conf_k = get_metric_statistics(np.array(dist_k_list), args.replication_times)
    
    print("-" * 30)
    print(f"Ground Truth Final Results:")
    print(f"Dist_g (Geometric): {mean_g:.4f} ± {conf_g:.4f}")
    print(f"Dist_k (Kinetic)  : {mean_k:.4f} ± {conf_k:.4f}")
    print("-" * 30)


def evaluate_dist_metrics(model, diffusion, loader, num_samples_limit, scale, device):
    """
    生成動作並計算 Dist_k 與 Dist_g
    """
    all_geometric_feats = []
    all_kinetic_feats = []
    
    #print(f'Generating motions for evaluation (limit: {num_samples_limit})...')
    
    generated_count = 0
    # 為了還原 3D 骨架，我們需要 mean 和 std (從 loader 的 dataset 取得)
    dataset = loader.dataset
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if generated_count >= num_samples_limit:
                break
                
            if isinstance(batch, tuple) and len(batch) == 2:
                # 這是 t2m_collate 的輸出格式: (motion, cond)
                # cond 結構: {'y': {'mask': ..., 'lengths': ..., 'text': ..., 'tokens': ...}}
                _, model_kwargs = batch
                m_lens = model_kwargs['y']['lengths']
                batch_size = m_lens.shape[0]
            elif isinstance(batch, list) or (isinstance(batch, tuple) and len(batch) >= 7):
                if len(batch) == 8:
                     word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _ = batch
                else:
                     word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _ = batch

                batch_size = motions.shape[0]

                # 手動建構 model_kwargs
                model_kwargs = {'y': {}}
                # 建立 mask: (Batch, 1, 1, Frames)
                max_frames = motions.shape[-1]
                mask = torch.arange(max_frames, device=device).expand(batch_size, max_frames) < m_lens.view(-1, 1).to(device)
                model_kwargs['y']['mask'] = mask.unsqueeze(1).unsqueeze(1)
                model_kwargs['y']['lengths'] = m_lens.to(device)
                model_kwargs['y']['text'] = caption # list of strings

            else:
                raise ValueError(f"Unexpected batch format with length {len(batch)}")

            # 確保 model_kwargs 中的 Tensor 都在正確的 device 上
            model_kwargs['y']['scale'] = torch.ones(batch_size, device=device) * scale
            if torch.is_tensor(model_kwargs['y']['lengths']):
                model_kwargs['y']['lengths'] = model_kwargs['y']['lengths'].to(device)
            if torch.is_tensor(model_kwargs['y']['mask']):
                model_kwargs['y']['mask'] = model_kwargs['y']['mask'].to(device)

            # 這裡我們使用 batch 中的最大長度
            max_frames = model_kwargs['y']['lengths'].max().item()
            # 執行採樣 (Sampling)
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                model,
                (batch_size, model.njoints, 1, max_frames), # shape
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            
            
            # sample shape: (Batch, 263, 1, Frames)
            sample = sample.squeeze(2).permute(0, 2, 1).cpu()

            # 2. 獲取並修正 Mean/Std 形狀 (解決 AIST++ 可能存在的 (263, 1) 形狀問題)
            mean = dataset.t2m_dataset.mean
            std = dataset.t2m_dataset.std
            # 如果是 numpy array 且有多餘的維度，將其移除
            if mean.ndim > 1: mean = mean.squeeze()
            if std.ndim > 1: std = std.squeeze()

            # 3. 手動執行反標準化 (不使用 dataset.inv_transform 以避免隱藏的廣播錯誤)
            sample = sample * std + mean
            sample = sample.float()

            # 4. 確保形狀正確後，傳入 recover_from_ric
            #    此時 sample 必須是 (Batch, Frames, 263)
            n_joints = 22 if sample.shape[-1] == 263 else 21
            sample_xyz = recover_from_ric(sample, n_joints)
            
            sample_xyz = sample_xyz.numpy()
            
            # 提取特徵
            g_feat, k_feat = extract_features(sample_xyz)
            
            all_geometric_feats.append(g_feat)
            all_kinetic_feats.append(k_feat)
            
            generated_count += batch_size
            #print(f"Generated {generated_count}/{num_samples_limit} motions")

    # 合併所有 batch
    all_geometric_feats = np.concatenate(all_geometric_feats, axis=0)[:num_samples_limit]
    all_kinetic_feats = np.concatenate(all_kinetic_feats, axis=0)[:num_samples_limit]

    return all_geometric_feats, all_kinetic_feats


def evaluation(loader, model, diffusion, args, log_file):
    
    with open(log_file, 'w') as f:
        print(f"Start Evaluation: Dist_k and Dist_g", file=f, flush=True)
        
        dist_k_list = []
        dist_g_list = []
        
        replication_times = args.replication_times
        
        for rep in range(replication_times):
            print(f"Replication {rep+1}/{replication_times}")
            print(f"Replication {rep+1}/{replication_times}", file=f, flush=True)
            
            # 生成動作
            g_feats, k_feats = evaluate_dist_metrics(
                model, diffusion, loader, 
                num_samples_limit=args.num_samples_limit, 
                scale=args.guidance_param, 
                device=dist_util.dev()
            )
            
            # 計算 Diversity
            # diversity_times 通常設為樣本數，或者像 AIST++ 論文設定 (例如 100)
            # 這裡我們使用全量計算
            div_times = args.diversity_times if args.diversity_times else len(g_feats)
            
            dist_g = calculate_diversity(g_feats, div_times)
            dist_k = calculate_diversity(k_feats, div_times)
            
            dist_g_list.append(dist_g)
            dist_k_list.append(dist_k)
            
            print(f"  Dist_g: {dist_g:.4f}")
            print(f"  Dist_k: {dist_k:.4f}")
            print(f"  Dist_g: {dist_g:.4f}", file=f, flush=True)
            print(f"  Dist_k: {dist_k:.4f}", file=f, flush=True)

        # 統計結果
        mean_g, conf_g = get_metric_statistics(np.array(dist_g_list), replication_times)
        mean_k, conf_k = get_metric_statistics(np.array(dist_k_list), replication_times)
        
        print("="*30)
        print(f"Final Results ({replication_times} replications):")
        print(f"Dist_g (Geometric Spread): {mean_g:.4f} ± {conf_g:.4f}")
        print(f"Dist_k (Kinetic Spread)  : {mean_k:.4f} ± {conf_k:.4f}")
        
        print("="*30, file=f, flush=True)
        print(f"Final Results ({replication_times} replications):", file=f, flush=True)
        print(f"Dist_g (Geometric Spread): {mean_g:.4f} ± {conf_g:.4f}", file=f, flush=True)
        print(f"Dist_k (Kinetic Spread)  : {mean_k:.4f} ± {conf_k:.4f}", file=f, flush=True)


if __name__ == '__main__':
    args = evaluation_parser()
    fixseed(args.seed)
    
    # 設置一些參數
    # args.batch_size 可以調大一點以加速生成
    args.replication_times = 20 
    args.num_samples_limit = 100 # 預設生成 100 個樣本 (根據您的 AIST++ 測試集大小調整)
    args.diversity_times = 100
    
    # 處理 Log 檔名
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), f'eval_unimumo_dist_{name}_{niter}.log')

    dist_util.setup_dist(args.device)
    
    print("Creating data loader...")
    # 注意：這裡使用 get_dataset_loader，與 eval_humanml 類似
    # 確保 split='test'
    loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=None, split='test', hml_mode='eval', data_dir=args.data_dir)

    print("Creating model...")
    model, diffusion = create_model_and_diffusion(args, loader)
    
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()


    evaluate_ground_truth(loader, args)
    evaluation(loader, model, diffusion, args, log_file)
