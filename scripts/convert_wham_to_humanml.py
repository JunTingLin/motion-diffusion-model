"""
WHAM to HumanML3D Format Converter

This script converts WHAM output (SMPL format) to HumanML3D format (263-dim features).

Usage:
    # 單一影片轉換
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

Requirements:
    - WHAM output pkl files
    - SMPL model file
    - MDM environment with dependencies
"""

import os
import sys
import argparse
import numpy as np
import torch
import joblib
from tqdm import tqdm
from os.path import join as pjoin

# Add the parent directory to path to import MDM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.humanml.scripts.motion_process import extract_features
from data_loaders.humanml.utils.paramUtil import (
    t2m_kinematic_chain,
    t2m_raw_offsets,
)

# FPS constant for HumanML3D
HUMANML_FPS = 20


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


def smpl_to_joints(smpl_model, smpl_poses, smpl_trans, target_fps=20, source_fps=30):
    """
    Convert SMPL parameters to joint positions.

    Args:
        smpl_model: SMPL model instance
        smpl_poses: (N, 72) SMPL pose parameters
        smpl_trans: (N, 3) root translation (in meters)
        target_fps: Target FPS for output (default: 20 for HumanML3D)
        source_fps: Source FPS of video (default: 30)

    Returns:
        joints: (M, 22, 3) joint positions in HumanML3D format
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smpl_model = smpl_model.to(device)

    # Resample from source_fps to target_fps
    n_frames_original = smpl_poses.shape[0]

    if source_fps != target_fps:
        # Use linear interpolation for resampling
        duration = n_frames_original / source_fps
        n_frames_target = int(duration * target_fps)

        # Create interpolation indices
        original_times = np.linspace(0, 1, n_frames_original)
        target_times = np.linspace(0, 1, n_frames_target)

        # Interpolate poses and trans
        from scipy.interpolate import interp1d

        pose_interp = interp1d(original_times, smpl_poses, axis=0, kind='linear')
        trans_interp = interp1d(original_times, smpl_trans, axis=0, kind='linear')

        smpl_poses = pose_interp(target_times)
        smpl_trans = trans_interp(target_times)

        print(f"  Resampled: {n_frames_original} frames ({source_fps} FPS) → {len(smpl_poses)} frames ({target_fps} FPS)")

    n_frames = smpl_poses.shape[0]
    all_joints = []

    # Process in batches to avoid OOM
    batch_size = 64
    for i in range(0, n_frames, batch_size):
        end_idx = min(i + batch_size, n_frames)
        batch_poses = torch.from_numpy(smpl_poses[i:end_idx]).float().to(device)
        batch_trans = torch.from_numpy(smpl_trans[i:end_idx]).float().to(device)

        # WHAM output is already in meters, no scaling needed

        # SMPL forward pass
        output = smpl_model(
            body_pose=batch_poses[:, 3:],      # (B, 69)
            global_orient=batch_poses[:, :3],  # (B, 3)
            transl=batch_trans,                # (B, 3)
            return_verts=False
        )

        # SMPL returns 24 joints, we need HumanML3D's 22 joints
        joints_24 = output.joints.detach().cpu().numpy()  # (B, 24, 3)
        joints_22 = joints_24[:, :22, :]  # (B, 22, 3)

        all_joints.append(joints_22)

    joints = np.concatenate(all_joints, axis=0)  # (N, 22, 3)
    return joints


def convert_wham_motion(wham_pkl_path, smpl_model, output_dir, motion_name, source_fps=30, subject_id=None):
    """
    Convert a single WHAM output to HumanML3D format.

    Args:
        wham_pkl_path: Path to WHAM output pkl file
        smpl_model: SMPL model instance
        output_dir: Output directory for HumanML3D format
        motion_name: Name of the motion (without extension)
        source_fps: FPS of the source video
        subject_id: Subject ID to extract (if None, uses first subject or single result)

    Returns:
        success: True if conversion succeeded
    """
    try:
        # Load WHAM output (WHAM uses joblib to save pkl files)
        wham_data = joblib.load(wham_pkl_path)

        # Handle different WHAM output structures
        # WHAM can output dict with subject IDs as keys, or direct results
        if isinstance(wham_data, dict):
            # Check if it's a multi-subject output (keys are integers/numpy integers)
            # Use np.issubdtype to handle both int and np.int64
            keys = list(wham_data.keys())
            if all(isinstance(k, (int, np.integer)) for k in keys):
                # Multi-subject: select one
                if subject_id is not None:
                    if subject_id not in wham_data:
                        print(f"  Warning: Subject {subject_id} not found, using first subject")
                        subject_id = keys[0]
                else:
                    subject_id = keys[0]
                results = wham_data[subject_id]
                print(f"  Using subject ID: {subject_id}")
            else:
                # Single result with named keys
                results = wham_data
        else:
            results = wham_data

        # Extract SMPL parameters
        # WHAM uses 'pose_world' and 'trans_world' for world coordinates
        # Fall back to 'pose' and 'trans' if world coords not available
        if 'pose_world' in results:
            smpl_poses = np.array(results['pose_world'])  # (N, 72)
            smpl_trans = np.array(results['trans_world'])  # (N, 3)
            print(f"  Using world coordinates")
        else:
            smpl_poses = np.array(results['pose'])  # (N, 72)
            smpl_trans = np.array(results['trans'])  # (N, 3)
            print(f"  Using camera coordinates (world coords not available)")

        n_frames = smpl_poses.shape[0]
        print(f"  Input: {n_frames} frames")

        # Convert SMPL to joint positions with FPS resampling
        joints = smpl_to_joints(
            smpl_model, smpl_poses, smpl_trans,
            target_fps=HUMANML_FPS, source_fps=source_fps
        )

        print(f"  Joints shape: {joints.shape}")

        # Extract HumanML3D features (263-dim)
        face_joint_indx = [2, 1, 17, 16]  # [right_hip, left_hip, right_shoulder, left_shoulder]
        fid_r = [8, 11]  # right ankle, right foot
        fid_l = [7, 10]  # left ankle, left foot

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

        print(f"  Features shape: {humanml_features.shape}")

        # Save outputs
        # 1. Joint positions (for visualization)
        joints_dir = pjoin(output_dir, 'new_joints')
        os.makedirs(joints_dir, exist_ok=True)
        np.save(pjoin(joints_dir, f'{motion_name}.npy'), joints)

        # 2. HumanML3D features (for model input)
        features_dir = pjoin(output_dir, 'new_joint_vecs')
        os.makedirs(features_dir, exist_ok=True)
        np.save(pjoin(features_dir, f'{motion_name}.npy'), humanml_features)

        # 3. Create a simple text file (required by MDM data loader)
        # Format: caption#word/POS word/POS ...#start_time#end_time
        texts_dir = pjoin(output_dir, 'texts')
        os.makedirs(texts_dir, exist_ok=True)
        with open(pjoin(texts_dir, f'{motion_name}.txt'), 'w') as f:
            f.write(f"a person performing motion#person/NOUN performing/VERB motion/NOUN#0.0#0.0\n")

        return True

    except Exception as e:
        print(f"Error converting {motion_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def create_file_list(output_dir, motion_names):
    """Create file list for MDM data loader."""
    # Write all.txt (used for in-between task)
    with open(pjoin(output_dir, 'all.txt'), 'w') as f:
        f.write('\n'.join(motion_names) + '\n')

    # Also create test.txt for evaluation
    with open(pjoin(output_dir, 'test.txt'), 'w') as f:
        f.write('\n'.join(motion_names) + '\n')

    print(f"\nCreated file lists:")
    print(f"  all.txt: {len(motion_names)} samples")
    print(f"  test.txt: {len(motion_names)} samples")


def calculate_statistics(output_dir, use_humanml_stats=True):
    """
    Calculate or copy mean and std for normalization.

    Args:
        output_dir: Output directory
        use_humanml_stats: If True, use HumanML3D pretrained stats (recommended)
    """
    if use_humanml_stats:
        # Use HumanML3D statistics for compatibility with pretrained model
        humanml_dir = pjoin(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'dataset', 'HumanML3D')

        if os.path.exists(pjoin(humanml_dir, 'Mean.npy')):
            import shutil
            shutil.copy(pjoin(humanml_dir, 'Mean.npy'), pjoin(output_dir, 'Mean.npy'))
            shutil.copy(pjoin(humanml_dir, 'Std.npy'), pjoin(output_dir, 'Std.npy'))
            print(f"\nCopied HumanML3D statistics (recommended for pretrained models)")
            return
        else:
            print(f"Warning: HumanML3D stats not found at {humanml_dir}, calculating from data")

    # Calculate from converted data
    features_dir = pjoin(output_dir, 'new_joint_vecs')

    all_features = []
    for npy_file in os.listdir(features_dir):
        if npy_file.endswith('.npy'):
            features = np.load(pjoin(features_dir, npy_file))
            all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)

    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    np.save(pjoin(output_dir, 'Mean.npy'), mean)
    np.save(pjoin(output_dir, 'Std.npy'), std)

    print(f"\nCalculated statistics from converted data:")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape:  {std.shape}")
    print(f"  Total frames: {all_features.shape[0]}")


def main():
    parser = argparse.ArgumentParser(description='Convert WHAM output to HumanML3D format')
    parser.add_argument('--wham_pkl', type=str, default=None,
                        help='Path to single WHAM output pkl file')
    parser.add_argument('--wham_dir', type=str, default=None,
                        help='Path to directory containing multiple WHAM output pkl files')
    parser.add_argument('--output_dir', type=str, default='./dataset/custom',
                        help='Output directory for HumanML3D format data')
    parser.add_argument('--smpl_model_path', type=str,
                        default='./body_models/smpl/SMPL_NEUTRAL.pkl',
                        help='Path to SMPL model file')
    parser.add_argument('--source_fps', type=int, default=30,
                        help='FPS of source video (default: 30)')
    parser.add_argument('--use_own_stats', action='store_true',
                        help='Calculate statistics from converted data instead of using HumanML3D stats')
    args = parser.parse_args()

    # Validate inputs
    if args.wham_pkl is None and args.wham_dir is None:
        print("ERROR: Must specify either --wham_pkl or --wham_dir")
        sys.exit(1)

    if args.wham_pkl is not None and not os.path.exists(args.wham_pkl):
        print(f"ERROR: WHAM pkl file not found: {args.wham_pkl}")
        sys.exit(1)

    if args.wham_dir is not None and not os.path.exists(args.wham_dir):
        print(f"ERROR: WHAM directory not found: {args.wham_dir}")
        sys.exit(1)

    if not os.path.exists(args.smpl_model_path):
        print(f"ERROR: SMPL model not found: {args.smpl_model_path}")
        print(f"Please download SMPL model from https://smpl.is.tue.mpg.de/")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = load_smpl_model(args.smpl_model_path)

    # Get list of pkl files to convert
    if args.wham_pkl:
        pkl_files = [(args.wham_pkl, os.path.splitext(os.path.basename(args.wham_pkl))[0])]
    else:
        pkl_files = []
        for f in os.listdir(args.wham_dir):
            if f.endswith('.pkl'):
                pkl_files.append((pjoin(args.wham_dir, f), os.path.splitext(f)[0]))

    print(f"\nFound {len(pkl_files)} WHAM output files to convert")

    # Convert each motion
    successful_names = []
    print("\nConverting motions...")
    for pkl_path, motion_name in tqdm(pkl_files):
        print(f"\nProcessing: {motion_name}")
        if convert_wham_motion(pkl_path, smpl_model, args.output_dir, motion_name,
                               source_fps=args.source_fps):
            successful_names.append(motion_name)

    print(f"\n{'='*50}")
    print(f"Successfully converted {len(successful_names)}/{len(pkl_files)} motions")

    if successful_names:
        # Create file lists
        create_file_list(args.output_dir, successful_names)

        # Copy or calculate statistics
        calculate_statistics(args.output_dir, use_humanml_stats=not args.use_own_stats)

        print(f"\n✓ Conversion complete!")
        print(f"  Output directory: {args.output_dir}")
        print(f"  Joint positions: {pjoin(args.output_dir, 'new_joints')}")
        print(f"  HumanML3D features: {pjoin(args.output_dir, 'new_joint_vecs')}")
        print(f"\nNext steps:")
        print(f"  Use these files with MDM's in-between mode:")
        print(f"  python -m sample.edit --model_path ./save/humanml_trans_enc_512/model000475000.pt \\")
        print(f"      --edit_mode in_between --data_dir {args.output_dir}")


if __name__ == '__main__':
    main()
