import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a motion sequence and export it to MP4.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion_npy", type=str,
                        help="Path to a .npy file that already stores joint positions "
                             "with shape [num_frames, num_joints, 3].")
    source.add_argument("--results_path", type=str,
                        help="Path to a results.npy (output of sample.edit/generate).")
    parser.add_argument("--motion_name", type=str,
                        help="Motion name to visualize when --results_path is used.")
    parser.add_argument("--motion_index", type=int, default=0,
                        help="Index to pick when --results_path is used and --motion_name is omitted.")
    parser.add_argument("--dataset", type=str, default="humanml",
                        choices=["humanml", "kit"],
                        help="Dataset name for skeleton definition.")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Source motion FPS (HumanML3D uses 20).")
    parser.add_argument("--vis_fps", type=float, default=20.0,
                        help="FPS for the rendered video.")
    parser.add_argument("--start_sec", type=float, default=0.0,
                        help="Starting point in seconds.")
    parser.add_argument("--duration_sec", type=float, default=-1.0,
                        help="Duration in seconds to visualize. Use -1 for full length.")
    parser.add_argument("--output", type=str, default="visualization.mp4",
                        help="Output MP4 path.")
    return parser.parse_args()


def load_motion_from_results(results_path, motion_name, motion_index):
    data = np.load(results_path, allow_pickle=True).item()
    names = data.get("motion_names")
    idx = motion_index
    if motion_name:
        assert motion_name in names, f"Motion name [{motion_name}] not found in results file."
        idx = names.index(motion_name)
    motion = data["motion"][idx]
    # results.npy stores motions as [num_joints, 3, num_frames]
    if motion.ndim == 3 and motion.shape[0] in (21, 22, 25):
        motion = np.transpose(motion, (2, 0, 1))
    return motion, names[idx]


def load_motion(args):
    if args.motion_npy:
        motion = np.load(args.motion_npy)
        assert motion.ndim == 3, "Expected motion array of shape [frames, joints, 3]."
        name = os.path.splitext(os.path.basename(args.motion_npy))[0]
    else:
        motion, name = load_motion_from_results(args.results_path,
                                                args.motion_name,
                                                args.motion_index)
    return motion, name


def crop_motion(motion, fps, start_sec, duration_sec):
    start_idx = max(0, int(round(start_sec * fps)))
    end_idx = motion.shape[0] if duration_sec <= 0 else start_idx + int(round(duration_sec * fps))
    end_idx = min(end_idx, motion.shape[0])
    if start_idx >= end_idx:
        raise ValueError("Invalid crop parameters lead to empty motion segment.")
    return motion[start_idx:end_idx]


def get_skeleton(dataset):
    if dataset == "kit":
        return paramUtil.kit_kinematic_chain
    return paramUtil.t2m_kinematic_chain


def main():
    args = parse_args()
    motion, name = load_motion(args)
    motion = crop_motion(motion, args.fps, args.start_sec, args.duration_sec)
    skeleton = get_skeleton(args.dataset)

    title = f"{name} ({motion.shape[0] / args.fps:.2f}s)"
    ani = plot_3d_motion(
        save_path=args.output,
        kinematic_tree=skeleton,
        joints=motion,
        dataset=args.dataset,
        title=title,
        fps=args.vis_fps
    )
    ani.duration = motion.shape[0] / args.vis_fps
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ani.write_videofile(args.output, fps=args.vis_fps, codec='libx264', verbose=False, logger=None)
    print(f"[Done] Saved visualization to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
