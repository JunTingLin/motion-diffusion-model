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
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

---

## 2. 取得資料集

### 2-1. 下載已轉換好的 HumanML3D AIST++ dataset（推薦）

```bash
gdown "https://drive.google.com/uc?export=download&confirm=pbef&id=1jGRaMjh6v2AMyOA8OYHxQmzVm0DC27q1" -O AIST++.zip
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
gdown "https://drive.google.com/uc?export=download&confirm=pbef&id=1MY5tW0LbF2LJIgEWGxYu4uML2Ef-fub4" -O aist_mdm_50steps.zip
unzip aist_mdm_50steps.zip -d save/
rm aist_mdm_50steps.zip
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

#### 兩段影片的 In-Between 生成

如果想要將**兩段不同的動作**（例如兩支舞蹈影片）串接起來，並生成中間的過渡動作，請使用以下流程：

**Step 1: 將影片轉換為 HumanML3D 格式**

首先使用 [WHAM](https://github.com/JunTingLin/WHAM.git) 從影片中提取 SMPL 參數，然後轉換為 HumanML3D 格式：

```bash
# 將 WHAM 輸出轉換為 HumanML3D 格式
python scripts/convert_wham_to_humanml.py \
    --wham_pkl /path/to/wham_output.pkl \
    --output_dir ./dataset/custom \
    --smpl_model_path ./body_models/smpl/SMPL_NEUTRAL.pkl \
    --source_fps 30

# 批次轉換（整個資料夾）
python scripts/convert_wham_to_humanml.py \
    --wham_dir /path/to/wham_outputs/ \
    --output_dir ./dataset/custom \
    --smpl_model_path ./body_models/smpl/SMPL_NEUTRAL.pkl \
    --source_fps 30
```

**Step 2: 準備兩段動作的 In-Between 輸入**

```bash
python scripts/prepare_two_motion_inbetween.py \
    --motion_a ./dataset/custom/new_joint_vecs/video_A.npy \
    --motion_b ./dataset/custom/new_joint_vecs/video_B.npy \
    --output_dir ./dataset/inbetween_task \
    --frames_from_a 40 \
    --frames_from_b 40 \
    --transition_frames 60
```

| 參數 | 說明 |
|-----|------|
| `--frames_from_a` | 從影片 A 結尾取多少幀（40 幀 = 2 秒） |
| `--frames_from_b` | 從影片 B 開頭取多少幀（40 幀 = 2 秒） |
| `--transition_frames` | 要生成的過渡幀數（60 幀 = 3 秒） |

腳本會輸出 `prefix_end` 和 `suffix_start` 參數，用於下一步。

**Step 3: 執行 MDM In-Between 生成**

```bash
python -m sample.edit \
    --model_path ./save/aist_mdm_50steps/model000050000.pt \
    --edit_mode in_between \
    --data_dir ./dataset/inbetween_task \
    --prefix_end 0.2857 \
    --suffix_start 0.7143 \
    --num_samples 1 \
    --num_repetitions 3
```

> 注意：`prefix_end` 和 `suffix_start` 的值會根據 Step 2 的參數自動計算，請使用腳本輸出的值。

**Step 4: 合併完整動作並輸出影片**

```bash
python scripts/merge_inbetween_result.py \
    --motion_a ./dataset/custom/new_joints/video_A.npy \
    --motion_b ./dataset/custom/new_joints/video_B.npy \
    --results_npy ./save/aist_mdm_50steps/edit_xxx/results.npy \
    --output_dir ./results/full_transition \
    --frames_from_a 40 \
    --frames_from_b 40 \
    --rep_idx 0
```

| 參數 | 說明 |
|-----|------|
| `--motion_a` | 完整影片 A 的 joints（注意是 `new_joints/` 不是 `new_joint_vecs/`） |
| `--motion_b` | 完整影片 B 的 joints |
| `--results_npy` | MDM 生成的結果檔案 |
| `--rep_idx` | 選擇第幾個重複生成結果（0, 1, 2, ...） |

輸出檔案：
- `full_motion.npy`: 合併後的完整動作 [影片A] + [生成的過渡] + [影片B]
- `full_motion.mp4`: 視覺化影片（藍色=原始動作，橘色=生成的過渡）

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

> **⚠️ 評估指標說明**
>
> 目前的評估使用 HumanML3D 預訓練的 evaluator（`t2m/`）

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
