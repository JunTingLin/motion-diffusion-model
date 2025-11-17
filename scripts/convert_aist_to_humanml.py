"""
AIST++ to HumanML3D Format Converter

This script converts AIST++ SMPL motion data to HumanML3D format (263-dim features).

Usage:
    python scripts/convert_aist_to_humanml.py \
        --aist_dir /path/to/aist_plusplus_final/motions \
        --output_dir ./dataset/AIST++ \
        --smpl_model_path ./body_models/smpl/SMPL_NEUTRAL.pkl

Requirements:
    - AIST++ dataset with SMPL motion files (*.pkl)
    - SMPL model file
    - MDM environment with dependencies
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm
from os.path import join as pjoin

# Add the parent directory to path to import MDM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.humanml.scripts.motion_process import (
    process_file,
    extract_features
)
from data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
    t2m_tgt_skel_id
)
from data_loaders.humanml.common.skeleton import Skeleton
import torch as torch_humanml

# FPS constants
AIST_FPS = 60  # AIST++ is recorded at 60 FPS
HUMANML_FPS = 20  # HumanML3D uses 20 FPS


def load_smpl_model(model_path):
    """Load SMPL model for forward kinematics."""
    try:
        from smplx import SMPL
        smpl_model = SMPL(
            model_path=os.path.dirname(model_path),
            gender='neutral',
            batch_size=1
        )
        return smpl_model
    except ImportError:
        print("ERROR: smplx not installed. Please install with: pip install smplx")
        sys.exit(1)


def smpl_to_joints(smpl_model, smpl_poses, smpl_trans, smpl_scaling, target_fps=20, source_fps=60):
    """
    Convert SMPL parameters to joint positions.

    Args:
        smpl_model: SMPL model instance
        smpl_poses: (N, 72) SMPL pose parameters at source_fps
        smpl_trans: (N, 3) root translation at source_fps
        smpl_scaling: (1,) scaling factor
        target_fps: Target FPS for output (default: 20 for HumanML3D)
        source_fps: Source FPS of AIST++ data (default: 60)

    Returns:
        joints: (M, 22, 3) joint positions in HumanML3D format at target_fps
                where M = N * (target_fps / source_fps)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smpl_model.to(device)

    # Downsample from source_fps to target_fps
    # AIST++ is 60 FPS, HumanML3D is 20 FPS, so we keep every 3rd frame
    fps_ratio = source_fps / target_fps
    assert fps_ratio == int(fps_ratio), f"FPS ratio must be integer, got {fps_ratio}"
    downsample_step = int(fps_ratio)

    # Downsample the SMPL parameters
    smpl_poses = smpl_poses[::downsample_step]  # Keep every Nth frame
    smpl_trans = smpl_trans[::downsample_step]

    n_frames = smpl_poses.shape[0]
    all_joints = []

    # Process in batches to avoid OOM
    batch_size = 64
    for i in range(0, n_frames, batch_size):
        end_idx = min(i + batch_size, n_frames)
        batch_poses = torch.from_numpy(smpl_poses[i:end_idx]).float().to(device)
        batch_trans = torch.from_numpy(smpl_trans[i:end_idx]).float().to(device)

        # CRITICAL: AIST++ translation is in centimeters, need to convert to meters
        # The SMPL model outputs joints in meters, but adds translation directly
        # So we must convert translation from cm to m BEFORE passing to SMPL
        # smpl_scaling is typically ~90-100 (indicates cm scale)
        batch_trans = batch_trans / float(smpl_scaling)

        # SMPL forward pass
        # smpl_poses: (B, 72) = (B, 24*3) where 24 joints × 3 rotation params
        # First 3 params are global_orient, rest 69 are body_pose
        output = smpl_model(
            body_pose=batch_poses[:, 3:],      # (B, 69)
            global_orient=batch_poses[:, :3],  # (B, 3)
            transl=batch_trans,                # (B, 3) - now in meters!
            return_verts=False
        )

        # SMPL returns 24 joints, we need to map to HumanML3D's 22 joints
        # SMPL joint order: see https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py
        # HumanML3D uses 22 joints (subset of SMPL)
        joints_24 = output.joints.detach().cpu().numpy()  # (B, 24, 3) - already in meters!

        # Map SMPL 24 joints to HumanML3D 22 joints
        # HumanML3D excludes: left/right eye (23, 24 in SMPL indexing)
        # SMPL joints 0-21 map to HumanML3D 0-21
        joints_22 = joints_24[:, :22, :]  # (B, 22, 3)

        all_joints.append(joints_22)

    joints = np.concatenate(all_joints, axis=0)  # (N, 22, 3)
    return joints


def convert_aist_motion(aist_pkl_path, smpl_model, output_dir, motion_name, tgt_offsets):
    """
    Convert a single AIST++ motion file to HumanML3D format.

    Args:
        aist_pkl_path: Path to AIST++ motion pkl file
        smpl_model: SMPL model instance
        output_dir: Output directory for HumanML3D format
        motion_name: Name of the motion (without extension)
        tgt_offsets: Target skeleton offsets

    Returns:
        success: True if conversion succeeded
    """
    try:
        # Load AIST++ SMPL parameters
        with open(aist_pkl_path, 'rb') as f:
            motion_data = pickle.load(f)

        smpl_poses = motion_data['smpl_poses']    # (N, 72) at 60 FPS
        smpl_trans = motion_data['smpl_trans']    # (N, 3) at 60 FPS
        smpl_scaling = motion_data['smpl_scaling'] # (1,)

        n_frames_60fps = smpl_poses.shape[0]
        duration_sec = n_frames_60fps / AIST_FPS

        # Convert SMPL to joint positions with FPS downsampling
        joints = smpl_to_joints(
            smpl_model, smpl_poses, smpl_trans, smpl_scaling,
            target_fps=HUMANML_FPS, source_fps=AIST_FPS
        )

        n_frames_20fps = joints.shape[0]
        # Verify downsampling (allow ±1 frame tolerance for rounding)
        expected_frames = int(n_frames_60fps * HUMANML_FPS / AIST_FPS)
        assert abs(n_frames_20fps - expected_frames) <= 1, \
            f"Frame count mismatch: got {n_frames_20fps}, expected ~{expected_frames} (±1)"

        # Extract HumanML3D features (263-dim)
        # feet_thre: threshold for foot contact detection
        # Define HumanML3D specific parameters
        # face_joint_indx: indices for hip and shoulder joints used for rotation alignment
        # For HumanML3D (22 joints): [right_hip, left_hip, right_shoulder, left_shoulder]
        face_joint_indx = [2, 1, 17, 16]  # Based on HumanML3D joint ordering
        # foot indices for contact detection
        fid_r = [8, 11]  # right ankle, right foot
        fid_l = [7, 10]  # left ankle, left foot

        # Convert numpy arrays to torch tensors for extract_features
        # extract_features internally creates Skeleton which expects torch tensors
        n_raw_offsets_torch = torch.from_numpy(t2m_raw_offsets)

        humanml_features = extract_features(
            joints,
            feet_thre=0.002,
            n_raw_offsets=n_raw_offsets_torch,
            kinematic_chain=t2m_kinematic_chain,
            face_joint_indx=face_joint_indx,
            fid_r=fid_r,
            fid_l=fid_l
        )

        # Save outputs
        # 1. Joint positions (for visualization)
        joints_dir = pjoin(output_dir, 'new_joints')
        os.makedirs(joints_dir, exist_ok=True)
        np.save(pjoin(joints_dir, f'{motion_name}.npy'), joints)

        # 2. HumanML3D features (for training)
        features_dir = pjoin(output_dir, 'new_joint_vecs')
        os.makedirs(features_dir, exist_ok=True)
        np.save(pjoin(features_dir, f'{motion_name}.npy'), humanml_features)

        return True

    except Exception as e:
        print(f"Error converting {motion_name}: {str(e)}")
        return False


def create_split_files(output_dir, motion_names, train_ratio=0.8, val_ratio=0.1):
    """
    Create train/val/test split files.

    Args:
        output_dir: Output directory
        motion_names: List of motion names
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    n_total = len(motion_names)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Shuffle with fixed seed for reproducibility
    np.random.seed(42)
    shuffled_names = np.array(motion_names.copy())
    np.random.shuffle(shuffled_names)

    train_names = shuffled_names[:n_train]
    val_names = shuffled_names[n_train:n_train+n_val]
    test_names = shuffled_names[n_train+n_val:]

    # Write split files
    with open(pjoin(output_dir, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_names) + '\n')

    with open(pjoin(output_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_names) + '\n')

    with open(pjoin(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(test_names) + '\n')

    with open(pjoin(output_dir, 'all.txt'), 'w') as f:
        f.write('\n'.join(motion_names) + '\n')

    print(f"\nCreated splits:")
    print(f"  Train: {len(train_names)} samples")
    print(f"  Val:   {len(val_names)} samples")
    print(f"  Test:  {len(test_names)} samples")
    print(f"  Total: {len(motion_names)} samples")


def calculate_statistics(output_dir):
    """Calculate mean and std for normalization."""
    features_dir = pjoin(output_dir, 'new_joint_vecs')

    all_features = []
    for npy_file in os.listdir(features_dir):
        if npy_file.endswith('.npy'):
            features = np.load(pjoin(features_dir, npy_file))
            all_features.append(features)

    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)  # (total_frames, 263)

    # Calculate statistics
    mean = np.mean(all_features, axis=0)  # (263,)
    std = np.std(all_features, axis=0)    # (263,)

    # Save statistics
    np.save(pjoin(output_dir, 'Mean.npy'), mean)
    np.save(pjoin(output_dir, 'Std.npy'), std)

    print(f"\nCalculated statistics:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape:  {std.shape}")
    print(f"  Total frames: {all_features.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description='Convert AIST++ to HumanML3D format')
    parser.add_argument('--aist_dir', type=str, required=True,
                        help='Path to AIST++ motions directory')
    parser.add_argument('--output_dir', type=str, default='./dataset/AIST++',
                        help='Output directory for HumanML3D format data')
    parser.add_argument('--smpl_model_path', type=str,
                        default='./body_models/smpl/SMPL_NEUTRAL.pkl',
                        help='Path to SMPL model file')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to convert (for testing)')
    args = parser.parse_args()

    # Check inputs
    if not os.path.exists(args.aist_dir):
        print(f"ERROR: AIST++ directory not found: {args.aist_dir}")
        sys.exit(1)

    if not os.path.exists(args.smpl_model_path):
        print(f"ERROR: SMPL model not found: {args.smpl_model_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = load_smpl_model(args.smpl_model_path)

    # Load target skeleton offsets
    from data_loaders.humanml.utils.paramUtil import t2m_tgt_skel_id
    from data_loaders.humanml.common.skeleton import Skeleton

    # Initialize target skeleton (using reference skeleton from HumanML3D)
    tgt_skel = Skeleton(torch_humanml.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
    print(f"Target skeleton initialized with {len(t2m_kinematic_chain)} chains")

    # Get reference skeleton offsets (will be computed from data)
    tgt_offsets = torch_humanml.from_numpy(t2m_raw_offsets)

    # Get list of AIST++ motion files
    motion_files = [f for f in os.listdir(args.aist_dir) if f.endswith('.pkl')]
    if args.max_samples:
        motion_files = motion_files[:args.max_samples]

    print(f"\nFound {len(motion_files)} AIST++ motion files")

    # Convert each motion
    successful_names = []
    print("\nConverting motions...")
    for motion_file in tqdm(motion_files):
        motion_name = motion_file[:-4]  # Remove .pkl extension
        aist_pkl_path = pjoin(args.aist_dir, motion_file)

        if convert_aist_motion(aist_pkl_path, smpl_model, args.output_dir, motion_name, tgt_offsets):
            successful_names.append(motion_name)

    print(f"\nSuccessfully converted {len(successful_names)}/{len(motion_files)} motions")

    # Create split files
    if successful_names:
        create_split_files(args.output_dir, successful_names)

        # Calculate statistics
        print("\nCalculating dataset statistics...")
        calculate_statistics(args.output_dir)

        print(f"\n✓ Conversion complete!")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Joint positions: {pjoin(args.output_dir, 'new_joints')}")
        print(f"  HumanML3D features: {pjoin(args.output_dir, 'new_joint_vecs')}")


if __name__ == '__main__':
    main()
