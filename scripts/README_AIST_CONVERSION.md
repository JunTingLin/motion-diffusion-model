# AIST++ to HumanML3D Conversion - Complete Guide

## ✅ 腳本已修復完成！

`convert_aist_to_humanml.py` 現在可以完整轉換 AIST++ 數據到 HumanML3D 格式。

### 🔧 最新修復（2025-11-17）

1. **FPS Downsampling**: 正確將 60 FPS 降採樣至 20 FPS（每 3 幀取 1 幀）
2. **Unit Conversion**: 修正單位轉換順序（在 SMPL forward 之前將 cm 轉為 m）
3. **Frame Validation**: 允許 ±1 幀容差，避免四捨五入誤判

**重要：** 如果之前已轉換過資料，請重新執行轉換以套用這些修正！

---

## 快速開始

### 完整轉換（一鍵完成）

```bash
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++ \
    --smpl_model_path ./body_models/smpl/SMPL_NEUTRAL.pkl
```

**這會自動完成：**
- ✅ SMPL → Joint Positions (22 joints)
- ✅ Joint Positions → HumanML3D Features (263-dim)
- ✅ 計算 Mean.npy / Std.npy
- ✅ 創建 train/val/test 分割（80%/10%/10%）
- ✅ 生成檔案列表

**預估時間：** 約 30-40 分鐘（1,408 個樣本）

---

## 輸出結構

```
dataset/AIST++/
├── new_joints/              # 關節位置（用於可視化）
│   ├── gBR_*.npy           # (N, 22, 3)
│   └── ...
├── new_joint_vecs/          # HumanML3D 特徵（用於訓練）
│   ├── gBR_*.npy           # (N-1, 263)
│   └── ...
├── Mean.npy                 # 特徵均值 (263,)
├── Std.npy                  # 特徵標準差 (263,)
├── all.txt                  # 所有動作列表
├── train.txt                # 訓練集（80%）
├── val.txt                  # 驗證集（10%）
└── test.txt                 # 測試集（10%）
```

---

## 使用轉換後的數據

### 1. 訓練 MDM 模型

```bash
python -m train.train_mdm \
    --save_dir save/aist_mdm \
    --dataset humanml \
    --data_dir ./dataset/AIST++ \
    --num_epochs 1000
```

### 2. In-between 編輯

首先，修改 `dataset/AIST++/test.txt` 選擇想要編輯的動作：

```bash
# 編輯 dataset/AIST++/test.txt，例如只保留 4 個動作
# gBR_sBM_cAll_d04_mBR0_ch01
# gBR_sBM_cAll_d04_mBR0_ch02
# gLO_sBM_cAll_d14_mLO2_ch06
# gWA_sFM_cAll_d27_mWA0_ch01

# 刪除 cache 讓 data loader 重新讀取
rm -f dataset/AIST++/t2m_*.npy

# 執行 in-between 生成
python -m sample.edit \
    --model_path ./save/humanml_trans_enc_512/model000200000.pt \
    --edit_mode in_between \
    --prefix_end 0.15 \
    --suffix_start 0.75 \
    --process_all \
    --num_repetitions 1 \
    --dataset humanml \
    --data_dir ./dataset/AIST++
```

### 3. 生成新動作（Text-to-Motion）

```bash
python -m sample.generate \
    --model_path ./save/aist_mdm/model000200000.pt \
    --text_prompt "a person dancing energetically" \
    --num_samples 5 \
    --dataset humanml \
    --data_dir ./dataset/AIST++
```

---

## 命令行參數

### 必需參數

- `--aist_dir`: AIST++ motions 目錄路徑
  - 例如：`/path/to/aist_plusplus_final/motions`

### 可選參數

- `--output_dir`: 輸出目錄（默認：`./dataset/AIST++`）
- `--smpl_model_path`: SMPL 模型路徑（默認：`./body_models/smpl/SMPL_NEUTRAL.pkl`）
- `--max_samples`: 限制轉換數量（用於測試，默認：全部轉換）

---

## 測試轉換

在全量轉換前，建議先測試幾個樣本：

```bash
# 只轉換 5 個樣本測試
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++_test \
    --smpl_model_path ./body_models/smpl/SMPL_NEUTRAL.pkl \
    --max_samples 5

# 檢查輸出
ls -lh ./dataset/AIST++_test/
```

預期輸出：
```
✓ Successfully converted 5/5 motions

Created splits:
  Train: 4 samples
  Val:   0 samples
  Test:  1 samples
  Total: 5 samples

✓ Conversion complete!
```

---

## HumanML3D 263 維特徵詳解

每幀的 263 維特徵包含：

| 特徵類型 | 維度 | 說明 |
|---------|------|------|
| **Root data** | 4 | 根節點旋轉速度(1) + 線性速度(2) + 高度(1) |
| **Local positions** | 63 | 21 個關節的局部位置 (21×3) |
| **Joint rotations** | 126 | 21 個關節的 6D 連續旋轉 (21×6) |
| **Joint velocities** | 66 | 22 個關節的速度 (22×3) |
| **Foot contacts** | 4 | 左右腳、左右踝的接觸標籤 |
| **總計** | **263** | |

---

## 常見問題

### Q1: 轉換失敗怎麼辦？

**A:** 檢查以下幾點：
1. SMPL 模型是否存在：`./body_models/smpl/SMPL_NEUTRAL.pkl`
2. AIST++ 資料路徑是否正確
3. 環境是否正確激活：`.conda` 環境
4. 查看錯誤訊息，可能是記憶體不足（降低 `batch_size`）

### Q2: 為什麼 new_joint_vecs 的幀數是 N-1？

**A:** 因為特徵包含速度資訊，需要計算相鄰幀的差值：
- Joints: (720, 22, 3) → 720 幀
- Features: (719, 263) → 719 幀（少一幀用於計算速度）

### Q3: 可以只轉換部分動作嗎？

**A:** 可以！方法：
1. 創建一個包含所需動作 ID 的臨時目錄
2. 只複製需要的 `.pkl` 檔案到臨時目錄
3. 指定 `--aist_dir` 為臨時目錄

或使用 `--max_samples` 限制數量（但會按字母順序選擇）。

### Q4: Mean.npy 和 Std.npy 的作用？

**A:** 用於特徵歸一化：
```python
# 訓練時
normalized_features = (features - mean) / std

# 推理時
features = normalized_features * std + mean
```

這確保不同維度的特徵在相同尺度上，提升訓練效果。

### Q5: 可以合併 HumanML3D 和 AIST++ 訓練嗎？

**A:** 可以，但需要重新計算統計量：

```python
import numpy as np
import os

# 載入兩個資料集的特徵
humanml_dir = './dataset/HumanML3D/new_joint_vecs'
aist_dir = './dataset/AIST++/new_joint_vecs'

all_features = []
for dir_path in [humanml_dir, aist_dir]:
    for npy_file in os.listdir(dir_path):
        if npy_file.endswith('.npy'):
            features = np.load(os.path.join(dir_path, npy_file))
            all_features.append(features)

# 計算混合統計量
all_features = np.concatenate(all_features, axis=0)
combined_mean = np.mean(all_features, axis=0)
combined_std = np.std(all_features, axis=0)

# 儲存
np.save('./dataset/combined_mean.npy', combined_mean)
np.save('./dataset/combined_std.npy', combined_std)
```

---

## 進階選項

### 修改資料集分割比例

編輯 `scripts/convert_aist_to_humanml.py` 中的 `create_split_files` 函數：

```python
def create_split_files(output_dir, motion_names, train_ratio=0.8, val_ratio=0.1):
    # 修改 train_ratio 和 val_ratio
    # 例如：train_ratio=0.7, val_ratio=0.2 → 70% train, 20% val, 10% test
```

### 自訂關節映射

如果需要調整 SMPL 24 joints → HumanML3D 22 joints 的映射，修改：

```python
# scripts/convert_aist_to_humanml.py, line 96
joints_22 = joints_24[:, :22, :]  # 目前取前 22 個關節
```

---

## 轉換腳本

### convert_aist_to_humanml.py

- **功能**：完整轉換 SMPL → HumanML3D（一鍵完成）
- **輸出**：Joints + Features + Mean/Std + Train/Val/Test Splits
- **狀態**：✅ 已修復，測試通過
- **用途**：生產環境使用，推薦！

**特點：**
- 自動完成所有轉換步驟
- 生成訓練所需的完整數據
- 支持測試模式（`--max_samples`）

---

## 下一步

1. ✅ **全量轉換 AIST++**
   ```bash
   python scripts/convert_aist_to_humanml.py \
       --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
       --output_dir ./dataset/AIST++
   ```

2. ✅ **測試 in-between 編輯**
   ```bash
   python -m sample.edit --model_path ... --edit_mode in_between --process_all
   ```

3. ✅ **訓練 AIST++ 模型**（可選）
   ```bash
   python -m train.train_mdm --save_dir save/aist_mdm --data_dir ./dataset/AIST++
   ```

---

## 技術細節

### 轉換流程

```
AIST++ SMPL (.pkl)
    ↓
  smpl_poses (N, 72) @ 60 FPS
  smpl_trans (N, 3) @ 60 FPS (單位：厘米)
  smpl_scaling (1,)
    ↓
[FPS Downsampling: 60 FPS → 20 FPS]
  - 每 3 幀取 1 幀 (60/20 = 3)
  - smpl_poses: (N, 72) → (M, 72) where M = N/3
  - smpl_trans: (N, 3) → (M, 3) where M = N/3
    ↓
[Unit Conversion: cm → meters]
  - smpl_trans = smpl_trans / smpl_scaling
  - 將厘米轉換為公尺（SMPL 需要公尺單位）
    ↓
[SMPL Forward Kinematics]
  - 輸入：poses (M, 72), trans (M, 3) in meters
  - 輸出：joints_24 (M, 24, 3) in meters
    ↓
[取前 22 個關節]
    ↓
  joints_22 (M, 22, 3) @ 20 FPS
    ↓
[extract_features]
  - 計算旋轉（IK）
  - 計算速度
  - 檢測腳部接觸
    ↓
  HumanML3D features (M-1, 263) @ 20 FPS
```

### 重要修正說明

#### 1. FPS Downsampling（2025-11-17 修復）

**問題：**
- AIST++ 原始資料是 60 FPS
- HumanML3D 使用 20 FPS
- 之前的轉換保留了所有 60 FPS 的幀，導致播放速度慢 3 倍

**解決方案：**
- 在 SMPL forward pass 之前進行降採樣
- 每 3 幀取 1 幀（60/20 = 3）
- 實現位置：`smpl_to_joints()` 函數，第 81-89 行

**範例：**
```python
# 720 frames @ 60 FPS → 240 frames @ 20 FPS
# Duration: 12.0 seconds (保持不變)
downsample_step = int(60 / 20)  # = 3
smpl_poses = smpl_poses[::downsample_step]
smpl_trans = smpl_trans[::downsample_step]
```

#### 2. Unit Conversion（2025-11-17 修復）

**問題：**
- AIST++ 的 `smpl_trans` 使用厘米單位（~169 cm）
- `smpl_scaling` 是縮放因子（~92-93）
- SMPL 模型需要公尺單位
- 之前在 SMPL 輸出後才轉換，導致骨架位置錯誤（Y 軸在 15,000 而非 0-2 米）

**解決方案：**
- 在 SMPL forward pass **之前**轉換單位
- `smpl_trans = smpl_trans / smpl_scaling`
- 實現位置：`smpl_to_joints()` 函數，第 101-105 行

**關鍵代碼：**
```python
# CRITICAL: 必須在 SMPL forward 之前轉換！
batch_trans = batch_trans / float(smpl_scaling)  # cm → m
output = smpl_model(
    body_pose=batch_poses[:, 3:],
    global_orient=batch_poses[:, :3],
    transl=batch_trans,  # 已經是公尺單位
    return_verts=False
)
```

#### 3. Frame Count Validation（2025-11-17 修復）

**問題：**
- 當原始幀數不能被 3 整除時，會產生 ±1 的四捨五入差異
- 例如：640 frames → 640/3 = 213.33 → 實際 214 frames
- 嚴格的 `==` 驗證導致 693/1408 (49%) 轉換報錯

**解決方案：**
- 允許 ±1 幀的容差
- `abs(n_frames_20fps - expected_frames) <= 1`
- 實現位置：`convert_aist_motion()` 函數，第 168 行

### 關鍵函數

1. **smpl_to_joints()**: SMPL 參數 → 關節位置
2. **extract_features()**: 關節位置 → 263 維特徵
3. **create_split_files()**: 創建資料集分割
4. **calculate_statistics()**: 計算 Mean/Std

---

## 故障排除

### 錯誤：AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

**原因**：Skeleton 類期望 torch tensor，但接收到 numpy array

**解決**：已修復，`n_raw_offsets` 會自動轉換為 torch tensor

### 錯誤：CUDA out of memory

**解決**：降低 batch_size
```python
# scripts/convert_aist_to_humanml.py, line 57
batch_size = 32  # 原本是 64，改為 32 或 16
```

### 錯誤：FileNotFoundError: SMPL model not found

**解決**：確認 SMPL 模型路徑
```bash
ls -lh ./body_models/smpl/SMPL_NEUTRAL.pkl
```

---

## 聯絡與支援

如果遇到問題：
1. 檢查本文檔的「常見問題」和「故障排除」
2. 確認環境正確（`.conda` 環境）
3. 查看完整的錯誤訊息

祝轉換順利！🎉
