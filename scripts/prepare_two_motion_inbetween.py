"""
Prepare Two Motions for In-Between Generation

This script takes two motion sequences and prepares them for MDM's in-between mode.
It concatenates the end of motion A with the beginning of motion B, leaving a gap
in the middle for the model to generate a smooth transition.

Usage:
    python scripts/prepare_two_motion_inbetween.py \
        --motion_a ./dataset/custom/new_joint_vecs/IMG_2234.npy \
        --motion_b ./dataset/custom/new_joint_vecs/IMG_2235.npy \
        --output_dir ./dataset/inbetween_task \
        --frames_from_a 40 \
        --frames_from_b 40 \
        --transition_frames 60

This will create:
    - A combined motion with: [last 40 frames of A] + [60 blank frames] + [first 40 frames of B]
    - The model will generate the 60 transition frames
"""

import os
import sys
import argparse
import numpy as np
import shutil
from os.path import join as pjoin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_inbetween_motion(
    motion_a_path: str,
    motion_b_path: str,
    output_dir: str,
    frames_from_a: int = 40,
    frames_from_b: int = 40,
    transition_frames: int = 60,
    output_name: str = "transition"
):
    """
    Prepare two motions for in-between generation.

    Args:
        motion_a_path: Path to first motion's 263-dim features (.npy)
        motion_b_path: Path to second motion's 263-dim features (.npy)
        output_dir: Output directory
        frames_from_a: Number of frames to take from the END of motion A
        frames_from_b: Number of frames to take from the START of motion B
        transition_frames: Number of frames for the model to generate (transition length)
        output_name: Name for the output motion file
    """
    # Load motions
    motion_a = np.load(motion_a_path)  # (N, 263)
    motion_b = np.load(motion_b_path)  # (M, 263)

    print(f"Motion A: {motion_a.shape} ({motion_a.shape[0] / 20:.1f} sec)")
    print(f"Motion B: {motion_b.shape} ({motion_b.shape[0] / 20:.1f} sec)")

    # Validate
    if frames_from_a > motion_a.shape[0]:
        print(f"Warning: frames_from_a ({frames_from_a}) > motion A length ({motion_a.shape[0]})")
        frames_from_a = motion_a.shape[0]

    if frames_from_b > motion_b.shape[0]:
        print(f"Warning: frames_from_b ({frames_from_b}) > motion B length ({motion_b.shape[0]})")
        frames_from_b = motion_b.shape[0]

    # Extract segments
    segment_a = motion_a[-frames_from_a:]  # Last N frames of A
    segment_b = motion_b[:frames_from_b]   # First M frames of B

    # Create placeholder for transition (zeros or interpolation)
    # Using zeros - the model will generate these frames
    transition = np.zeros((transition_frames, 263), dtype=np.float32)

    # Concatenate: [segment_a] + [transition] + [segment_b]
    combined_motion = np.concatenate([segment_a, transition, segment_b], axis=0)

    total_frames = combined_motion.shape[0]
    print(f"\nCombined motion: {combined_motion.shape}")
    print(f"  - Frames from A (end):   {frames_from_a} frames ({frames_from_a/20:.1f} sec)")
    print(f"  - Transition (generate): {transition_frames} frames ({transition_frames/20:.1f} sec)")
    print(f"  - Frames from B (start): {frames_from_b} frames ({frames_from_b/20:.1f} sec)")
    print(f"  - Total:                 {total_frames} frames ({total_frames/20:.1f} sec)")

    # Calculate prefix_end and suffix_start for MDM
    prefix_end = frames_from_a / total_frames
    suffix_start = (frames_from_a + transition_frames) / total_frames

    print(f"\nMDM parameters:")
    print(f"  --prefix_end {prefix_end:.4f}")
    print(f"  --suffix_start {suffix_start:.4f}")

    # Create output directories
    os.makedirs(pjoin(output_dir, 'new_joint_vecs'), exist_ok=True)
    os.makedirs(pjoin(output_dir, 'texts'), exist_ok=True)

    # Save combined motion
    output_path = pjoin(output_dir, 'new_joint_vecs', f'{output_name}.npy')
    np.save(output_path, combined_motion)
    print(f"\nSaved: {output_path}")

    # Create text file (format: caption#word/POS word/POS ...#start#end)
    with open(pjoin(output_dir, 'texts', f'{output_name}.txt'), 'w') as f:
        f.write("a person transitioning between two motions#person/NOUN transitioning/VERB between/ADP two/NUM motions/NOUN#0.0#0.0\n")

    # Create file list
    with open(pjoin(output_dir, 'test.txt'), 'w') as f:
        f.write(f"{output_name}\n")
    with open(pjoin(output_dir, 'all.txt'), 'w') as f:
        f.write(f"{output_name}\n")

    # Copy Mean.npy and Std.npy from source directory or HumanML3D
    source_dir = os.path.dirname(os.path.dirname(motion_a_path))
    for stat_file in ['Mean.npy', 'Std.npy']:
        src = pjoin(source_dir, stat_file)
        dst = pjoin(output_dir, stat_file)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied: {stat_file}")

    # Save metadata for reference
    metadata = {
        'motion_a': motion_a_path,
        'motion_b': motion_b_path,
        'frames_from_a': frames_from_a,
        'frames_from_b': frames_from_b,
        'transition_frames': transition_frames,
        'total_frames': total_frames,
        'prefix_end': prefix_end,
        'suffix_start': suffix_start,
    }
    np.save(pjoin(output_dir, f'{output_name}_metadata.npy'), metadata)

    print(f"\n" + "="*60)
    print(f"Ready! Run in-between with:")
    print(f"="*60)
    print(f"""
python -m sample.edit \\
    --model_path ./save/humanml_trans_enc_512/model000475000.pt \\
    --edit_mode in_between \\
    --data_dir {output_dir} \\
    --prefix_end {prefix_end:.4f} \\
    --suffix_start {suffix_start:.4f} \\
    --num_samples 1 \\
    --num_repetitions 3
""")

    return prefix_end, suffix_start


def main():
    parser = argparse.ArgumentParser(description='Prepare two motions for in-between generation')
    parser.add_argument('--motion_a', type=str, required=True,
                        help='Path to first motion (263-dim .npy)')
    parser.add_argument('--motion_b', type=str, required=True,
                        help='Path to second motion (263-dim .npy)')
    parser.add_argument('--output_dir', type=str, default='./dataset/inbetween_task',
                        help='Output directory')
    parser.add_argument('--frames_from_a', type=int, default=40,
                        help='Number of frames from end of motion A (default: 40 = 2 sec)')
    parser.add_argument('--frames_from_b', type=int, default=40,
                        help='Number of frames from start of motion B (default: 40 = 2 sec)')
    parser.add_argument('--transition_frames', type=int, default=60,
                        help='Number of transition frames to generate (default: 60 = 3 sec)')
    parser.add_argument('--output_name', type=str, default='transition',
                        help='Output motion name')

    args = parser.parse_args()

    if not os.path.exists(args.motion_a):
        print(f"Error: Motion A not found: {args.motion_a}")
        sys.exit(1)
    if not os.path.exists(args.motion_b):
        print(f"Error: Motion B not found: {args.motion_b}")
        sys.exit(1)

    prepare_inbetween_motion(
        args.motion_a,
        args.motion_b,
        args.output_dir,
        args.frames_from_a,
        args.frames_from_b,
        args.transition_frames,
        args.output_name
    )


if __name__ == '__main__':
    main()
