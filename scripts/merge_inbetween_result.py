"""
Merge In-Between Result with Full Motions

After running MDM's in-between generation, use this script to create
the full video with: [Full A] + [Generated Transition] + [Full B]

Usage:
    python scripts/merge_inbetween_result.py \
        --motion_a ./dataset/custom/new_joints/IMG_2234.npy \
        --motion_b ./dataset/custom/new_joints/IMG_2235.npy \
        --results_npy ./save/.../results.npy \
        --output_dir ./results/full_transition \
        --frames_from_a 40 \
        --frames_from_b 40 \
        --rep_idx 0

This will create:
    - full_motion.npy: merged motion [A] + [generated middle] + [B]
    - full_motion.mp4: rendered video
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def merge_motions(
    motion_a_path: str,
    motion_b_path: str,
    results_npy_path: str,
    output_dir: str,
    frames_from_a: int = 40,
    frames_from_b: int = 40,
    rep_idx: int = 0,
    render_video: bool = True
):
    """
    Merge the full motions with the generated transition.

    Args:
        motion_a_path: Path to full motion A joints (N, 22, 3)
        motion_b_path: Path to full motion B joints (M, 22, 3)
        results_npy_path: Path to MDM results.npy
        output_dir: Output directory
        frames_from_a: Number of frames taken from end of A
        frames_from_b: Number of frames taken from start of B
        rep_idx: Which repetition to use (0, 1, 2, ...)
        render_video: Whether to render video
    """
    # Load full motions (joints format: N, 22, 3)
    motion_a = np.load(motion_a_path)  # (N, 22, 3)
    motion_b = np.load(motion_b_path)  # (M, 22, 3)

    print(f"Motion A (joints): {motion_a.shape} ({motion_a.shape[0]/20:.1f} sec)")
    print(f"Motion B (joints): {motion_b.shape} ({motion_b.shape[0]/20:.1f} sec)")

    # Load generated results
    results = np.load(results_npy_path, allow_pickle=True).item()
    # results['motion'] shape: (num_reps, 22, 3, num_frames)
    generated_all = results['motion']
    print(f"Generated results: {generated_all.shape} (reps, joints, xyz, frames)")

    # Get actual length (MDM may pad to fixed length)
    actual_length = results['lengths'][rep_idx] if 'lengths' in results else generated_all.shape[3]
    print(f"Actual length from results['lengths']: {actual_length} frames")

    # Select repetition and transpose to (frames, joints, xyz)
    generated = generated_all[rep_idx]  # (22, 3, frames)
    generated = generated.transpose(2, 0, 1)  # (frames, 22, 3)

    # Truncate to actual length (remove padding)
    if generated.shape[0] > actual_length:
        print(f"Truncating from {generated.shape[0]} to {actual_length} frames (removing padding)")
        generated = generated[:actual_length]

    print(f"Selected rep {rep_idx}: {generated.shape} ({generated.shape[0]/20:.1f} sec)")

    # Extract only the transition part from generated
    # Generated structure: [frames_from_a] + [transition] + [frames_from_b]
    #
    # IMPORTANT: The transition's first frame should smoothly connect from A's last frame,
    # and transition's last frame should smoothly connect to B's first frame.
    #
    # So the correct merge is:
    #   [A without last frames_from_a] + [generated result] + [B without first frames_from_b]
    #
    # This way:
    #   - A's ending (last frames_from_a frames) is replaced by generated prefix
    #   - B's beginning (first frames_from_b frames) is replaced by generated suffix
    #   - The transition in the middle connects them smoothly

    transition_frames = generated.shape[0] - frames_from_a - frames_from_b

    if transition_frames <= 0:
        print(f"Error: No transition frames! Generated has {generated.shape[0]} frames, "
              f"but frames_from_a={frames_from_a} + frames_from_b={frames_from_b} = {frames_from_a + frames_from_b}")
        return None

    print(f"Generated breakdown:")
    print(f"  Prefix (from A):  {frames_from_a} frames")
    print(f"  Transition:       {transition_frames} frames")
    print(f"  Suffix (from B):  {frames_from_b} frames")

    # Correct merge:
    # [A's beginning, excluding last frames_from_a] + [entire generated] + [B's end, excluding first frames_from_b]
    motion_a_prefix = motion_a[:-frames_from_a]  # A without last frames_from_a frames
    motion_b_suffix = motion_b[frames_from_b:]   # B without first frames_from_b frames

    full_motion = np.concatenate([motion_a_prefix, generated, motion_b_suffix], axis=0)

    # Calculate actual frame counts in merged motion
    a_prefix_len = len(motion_a_prefix)
    generated_len = generated.shape[0]
    b_suffix_len = len(motion_b_suffix)

    print(f"\n{'='*50}")
    print(f"Full merged motion: {full_motion.shape}")
    print(f"  A prefix:    {a_prefix_len:4d} frames ({a_prefix_len/20:5.1f} sec) [original A without last {frames_from_a}f]")
    print(f"  Generated:   {generated_len:4d} frames ({generated_len/20:5.1f} sec) [prefix({frames_from_a}f) + transition({transition_frames}f) + suffix({frames_from_b}f)]")
    print(f"  B suffix:    {b_suffix_len:4d} frames ({b_suffix_len/20:5.1f} sec) [original B without first {frames_from_b}f]")
    print(f"  ─────────────────────────────────────")
    print(f"  Total:       {full_motion.shape[0]:4d} frames ({full_motion.shape[0]/20:5.1f} sec)")
    print(f"{'='*50}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_npy = os.path.join(output_dir, 'full_motion.npy')
    np.save(output_npy, full_motion)
    print(f"\nSaved joints: {output_npy}")

    # Calculate gt_frames (original/constrained frames = blue, freely generated = orange)
    # Following MDM's convention: gt_frames are shown in BLUE, non-gt in ORANGE
    #
    # In full_motion:
    #   - A prefix (0 to a_prefix_len-1): original A -> BLUE
    #   - Generated prefix (a_prefix_len to a_prefix_len+frames_from_a-1): constrained to A -> BLUE
    #   - Transition (a_prefix_len+frames_from_a to a_prefix_len+frames_from_a+transition_frames-1): freely generated -> ORANGE
    #   - Generated suffix (... to a_prefix_len+generated_len-1): constrained to B -> BLUE
    #   - B suffix (a_prefix_len+generated_len to end): original B -> BLUE

    transition_start = a_prefix_len + frames_from_a
    transition_end = transition_start + transition_frames

    # gt_frames = everything EXCEPT the transition part
    gt_frames = list(range(transition_start)) + list(range(transition_end, full_motion.shape[0]))

    print(f"\nFrame coloring (matching MDM convention):")
    print(f"  Blue (GT/constrained):  frames 0-{a_prefix_len-1} (A prefix, original)")
    print(f"  Blue (GT/constrained):  frames {a_prefix_len}-{transition_start-1} (Generated prefix, constrained to A)")
    print(f"  Orange (Generated):     frames {transition_start}-{transition_end-1} (Transition, freely generated)")
    print(f"  Blue (GT/constrained):  frames {transition_end}-{a_prefix_len+generated_len-1} (Generated suffix, constrained to B)")
    print(f"  Blue (GT/constrained):  frames {a_prefix_len+generated_len}-{full_motion.shape[0]-1} (B suffix, original)")

    # Render video
    if render_video:
        output_mp4 = os.path.join(output_dir, 'full_motion.mp4')
        print(f"\nRendering video...")
        success = render_motion_video(
            full_motion, output_mp4, fps=20,
            gt_frames=gt_frames,
            title=f"Full Motion: A({a_prefix_len}f) + Gen({generated_len}f) + B({b_suffix_len}f)"
        )
        if success:
            print(f"Saved video: {output_mp4}")

    return full_motion


def render_motion_video(motion, output_path, fps=20, gt_frames=None, title="Full Motion with Transition"):
    """
    Render motion to video using the project's visualization tools.

    Args:
        motion: (N, 22, 3) joint positions
        output_path: Output video path
        fps: Frames per second
        gt_frames: List of frame indices that are ground truth (will be blue)
        title: Title for the video
    """
    try:
        from data_loaders.humanml.utils.plot_script import plot_3d_motion
        from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain

        n_frames = motion.shape[0]

        # plot_3d_motion expects (frames, joints, xyz) format
        ani = plot_3d_motion(
            output_path,
            t2m_kinematic_chain,
            motion,  # (N, 22, 3)
            dataset='humanml',
            title=title,
            fps=fps,
            gt_frames=gt_frames if gt_frames else [],
            figsize=(10, 10),
            radius=4
        )

        # Write video
        ani = ani.set_duration(n_frames / fps)
        ani.write_videofile(output_path, fps=fps, codec='libx264', verbose=False, logger=None)

        return True
    except Exception as e:
        print(f"Warning: Could not render video: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Merge in-between result with full motions')
    parser.add_argument('--motion_a', type=str, required=True,
                        help='Path to full motion A joints (.npy, shape: N,22,3)')
    parser.add_argument('--motion_b', type=str, required=True,
                        help='Path to full motion B joints (.npy, shape: M,22,3)')
    parser.add_argument('--results_npy', type=str, required=True,
                        help='Path to MDM results.npy')
    parser.add_argument('--output_dir', type=str, default='./results/full_transition',
                        help='Output directory')
    parser.add_argument('--frames_from_a', type=int, default=40,
                        help='Number of frames taken from end of A')
    parser.add_argument('--frames_from_b', type=int, default=40,
                        help='Number of frames taken from start of B')
    parser.add_argument('--rep_idx', type=int, default=0,
                        help='Which repetition to use (0, 1, 2, ...)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video rendering')

    args = parser.parse_args()

    merge_motions(
        args.motion_a,
        args.motion_b,
        args.results_npy,
        args.output_dir,
        args.frames_from_a,
        args.frames_from_b,
        args.rep_idx,
        render_video=not args.no_video
    )


if __name__ == '__main__':
    main()
