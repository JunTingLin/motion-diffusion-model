# MDM: Human Motion Diffusion Model (AIST++ 版本)

[![arXiv](https://img.shields.io/badge/arXiv-<2209.14916>-<COLOR>.svg)](https://arxiv.org/abs/2209.14916)

基於 [MDM (Human Motion Diffusion Model)](https://arxiv.org/abs/2209.14916) 的 AIST++ 舞蹈動作生成實作。

原始專案：[motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model)

---

## 環境需求

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU

---

## 1. 環境設定

安裝 ffmpeg：

```bash
sudo apt update
sudo apt install ffmpeg
```

建立 conda 環境：

```bash
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install 'transformers<4.30' 'tokenizers<0.13'
pip install blobfile smplx matplotlib moviepy==1.0.3 imageio imageio-ffmpeg spacy
```

下載相依檔案：

```bash
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

| 檔案 | 用途 |
|-----|------|
| `glove/` | 文字轉 word embedding（評估用）|
| `t2m/` | Text-Motion Matching 評估模型 |


---

## 2. 取得資料集

### 2-1. 下載已轉換好的 HumanML3D AIST++ dataset（推薦）

```bash
gdown 1jGRaMjh6v2AMyOA8OYHxQmzVm0DC27q1
unzip AIST++.zip -d dataset/
rm AIST++.zip
```

### 2-2. 自己轉換 HumanML3D AIST++ dataset

<details>
<summary><b>展開查看轉換步驟</b></summary>

#### 下載 AIST++ 原始資料

下載 SMPL motions：
- https://storage.cloud.google.com/aist_plusplus_public/20210308/motions.zip

#### 轉換 AIST++ SMPL → HumanML3D 格式

```bash
python scripts/convert_aist_to_humanml.py \
    --aist_dir /path/to/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++
```

轉換內容：
- 60 FPS → 20 FPS 降採樣
- SMPL parameters → 263-dim HumanML3D features
- 自動計算 Mean.npy / Std.npy
- 自動產生 train/val/test split

#### 生成文字描述

```bash
# 先檢查缺少的文本檔案（dry run）
python scripts/generate_aist_texts.py --dry_run

# 確認沒問題後，實際生成
python scripts/generate_aist_texts.py
```

</details>

### 視覺化動作（確認資料集是否正確）

```bash
python scripts/visualize_motion.py \
    --motion_npy dataset/AIST++/new_joints/gWA_sBM_cAll_d25_mWA3_ch06.npy \
    --duration_sec -1 \
    --output gWA_sBM_cAll_d25_mWA3_ch06.mp4
```

| 參數 | 說明 |
|-----|------|
| `--duration_sec -1` | 使用完整長度 |
| `--duration_sec 5` | 只取前 5 秒 |

---

## 3. 取得預訓練模型

下載在 AIST++ 上訓練 50k steps 的模型（Unconditional）：

```bash
mkdir -p save/aist_mdm_50steps
gdown 1zShG6zYDIy1gcAdOtECsnMOpVunArWdK -O save/aist_mdm_50steps/model000050000.pt
```

| 模型資訊 | |
|---------|---|
| 訓練步數 | 50,000 steps |
| Diffusion steps | 50 |
| 訓練模式 | Unconditional |
| 資料集 | AIST++ (1,408 samples) |

---

## 4. 使用說明

> **重要**: 如果 `dataset/AIST++/test.txt` 有更動，請先刪除 cache 檔案再執行：
> ```bash
> rm dataset/AIST++/dataset/t2m_test.npy
> ```

### Test Set 切換

根據不同的評估需求，切換 `test.txt`：

```bash
# 全部測試資料 (142 samples)
cp dataset/AIST++/test_origin.txt dataset/AIST++/test.txt

# 各舞風各一個 (10 samples)
cp dataset/AIST++/test_choose.txt dataset/AIST++/test.txt

# 特定舞風 (例如 House)
cp dataset/AIST++/test_house.txt dataset/AIST++/test.txt
```

---

### 4-1. Edit (In-between)

對測試集的動作進行 in-between 補間生成：

```bash
python -m sample.edit \
    --model_path save/aist_mdm_50steps/model000050000.pt \
    --edit_mode in_between \
    --num_repetitions 3 \
    --seed 42 \
    --prefix_end 0.10 \
    --suffix_start 0.90 \
    --guidance_param 1.0 \
    --process_all \
    --data_dir ./dataset/AIST++
```

| 參數 | 說明 |
|-----|------|
| `--prefix_end 0.10` | 保留前 10% 作為 prefix |
| `--suffix_start 0.90` | 保留後 10% 作為 suffix |
| `--process_all` | 處理所有測試資料 |

---

### 4-2. Generate

生成無條件動作（unconstrained generation）：

```bash
python -m sample.generate \
    --model_path save/aist_mdm_50steps/model000050000.pt \
    --seed 42 \
    --num_samples 3 \
    --num_repetitions 3 \
    --guidance_param 1.0 \
    --unconstrained \
    --data_dir ./dataset/AIST++
```

---

### 4-3. Evaluation

執行評估：

```bash
python -m eval.eval_humanml \
    --model_path save/aist_mdm_50steps/model000050000.pt \
    --eval_mode wo_mm \
    --guidance_param 1.0 \
    --data_dir ./dataset/AIST++ \
    --seed 42
```

| eval_mode | 說明 |
|-----------|------|
| `debug` | 快速測試 (5 replications) |
| `wo_mm` | 不含 MultiModality (20 replications) |
| `mm_short` | 含 MultiModality (5 replications) |

---

### 4-4. Train

訓練模型（log 輸出到 train.log）：

```bash
python -m train.train_mdm \
    --save_dir save/aist_mdm_50steps \
    --dataset humanml \
    --data_dir ./dataset/AIST++ \
    --batch_size 128 \
    --lr 2e-4 \
    --diffusion_steps 50 \
    --mask_frames \
    --use_ema \
    --unconstrained \
    --gen_guidance_param 1.0 \
    --eval_during_training \
    --eval_split val \
    --num_steps 50000 \
    --log_interval 100 \
    --save_interval 5000 \
    --train_platform_type TensorboardPlatform \
    --overwrite > train.log 2>&1
```

查看訓練過程：

```bash
tensorboard --logdir=save/aist_mdm_50steps
```

---
