# Scripts


## 1. Convert AIST++

Convert entire AIST++ dataset to HumanML3D format.

```bash
python scripts/convert_aist_to_humanml.py \
    --aist_dir /mnt/d/Code/PythonProjects/CVPDL2025/final/aist_plusplus_final/motions \
    --output_dir ./dataset/AIST++
```

## 2. Visualize Motion

Visualize a HumanML3D motion sequence and export to MP4.

```bash
python scripts/visualize_motion.py \
    --motion_npy dataset/AIST++/new_joints/gWA_sBM_cAll_d25_mWA3_ch06.npy \
    --duration_sec -1 \
    --output gWA_sBM_cAll_d25_mWA3_ch06.mp4
```
