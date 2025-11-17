# AIST++ 轉換快速開始

## 🚀 一鍵轉換（推薦）

```bash
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++
```

**這會自動生成：**
- ✅ `new_joints/` - 關節位置 (N, 22, 3)
- ✅ `new_joint_vecs/` - HumanML3D 特徵 (N-1, 263) ← **訓練用**
- ✅ `Mean.npy` / `Std.npy` - 歸一化參數
- ✅ `train.txt` / `val.txt` / `test.txt` - 資料集分割

**預估時間：** 30-40 分鐘（1,408 個樣本）

---

## 📝 使用轉換後的數據

### 方案 1：In-between 編輯（你目前的需求）

```bash
# 1. 編輯 test.txt 選擇要處理的動作
nano dataset/AIST++/test.txt
# 只保留幾個動作 ID，例如：
# gBR_sBM_cAll_d04_mBR0_ch01
# gLO_sBM_cAll_d14_mLO2_ch06

# 2. 刪除 cache
rm -f dataset/AIST++/t2m_*.npy

# 3. 執行 in-between
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

### 方案 2：訓練新模型（未來需求）

```bash
python -m train.train_mdm \
    --save_dir save/aist_mdm \
    --dataset humanml \
    --data_dir ./dataset/AIST++ \
    --num_epochs 1000
```

---

## ⚠️ 注意事項

1. **轉換前先測試**：
   ```bash
   # 只轉換 5 個樣本測試
   python scripts/convert_aist_to_humanml.py \
       --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
       --output_dir ./dataset/AIST++_test \
       --max_samples 5
   ```

2. **確認環境**：使用 `.conda` 環境

3. **硬碟空間**：
   - 原始 SMPL 數據：~200MB
   - 轉換後數據：~500MB
   - 總需求：~1GB

---

## 🎯 完整流程示例

```bash
# Step 1: 測試轉換（5 個樣本）
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++_test \
    --max_samples 5

# Step 2: 檢查輸出
ls -lh ./dataset/AIST++_test/

# Step 3: 確認無誤後，全量轉換
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++

# Step 4: 使用轉換後的數據
python -m sample.edit --model_path ... --data_dir ./dataset/AIST++ --process_all
```

---

詳細說明請見：[README_AIST_CONVERSION.md](./README_AIST_CONVERSION.md)
