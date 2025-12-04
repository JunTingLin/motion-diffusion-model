# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from sample.generate import save_multiple_samples, construct_template_variables
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil


def main():
    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = 120 # min(max_frames, int(args.motion_length*fps))

    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')

    # If process_all is True, process all samples in the test set
    if args.process_all:
        print('Processing ALL samples in the test set...')
        data = get_dataset_loader(name=args.dataset,
                                  batch_size=1,  # Process one at a time
                                  num_frames=max_frames,
                                  split='test',
                                  hml_mode='train',
                                  data_dir=args.data_dir,
                                  disable_offset_aug=True)  # Always start from the beginning
        args.num_samples = len(data.dataset)
        args.batch_size = 1
        # Get motion IDs - dataset is sorted by length, not test.txt order!
        # Due to DataLoader shuffle=True, we must collect both IDs and data in one pass
        motion_ids = []
        cached_data = []
        temp_iterator = iter(data)
        for _ in range(args.num_samples):
            input_motions_temp, model_kwargs_temp = next(temp_iterator)
            # Extract motion name from db_key field
            motion_id = model_kwargs_temp['y']['db_key'][0] if 'db_key' in model_kwargs_temp['y'] and model_kwargs_temp['y']['db_key'][0] is not None else f'sample_{_}'
            motion_ids.append(motion_id)
            cached_data.append((input_motions_temp, model_kwargs_temp))
        print(f'Total samples to process: {args.num_samples}')
        print(f'Motion IDs: {motion_ids}')
    else:
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
        # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # (specify through the --seed flag)
        args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
        data = get_dataset_loader(name=args.dataset,
                                  batch_size=args.batch_size,
                                  num_frames=max_frames,
                                  split='test',
                                  hml_mode='train',
                                  data_dir=args.data_dir,
                                  disable_offset_aug=True)  # Always start from the beginning
        motion_ids = None  # Will get IDs during sampling
        cached_data = None  # No cached data for non-process_all mode
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    # Only use ClassifierFreeSampleModel for conditional models
    if args.guidance_param != 1 and model.cond_mode != 'no_cond':
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    all_motions = []
    all_lengths = []
    all_text = []
    all_input_motions = []
    all_gt_frames = []
    all_motion_names = []

    # Process each sample in the dataset (in data loader's sorted order)
    if cached_data is not None:
        # Use cached data (process_all mode with shuffle=True)
        data_source = cached_data
    else:
        # Use iterator (non-process_all mode)
        data_source = iter(data)

    for sample_idx in range(args.num_samples):
        if args.process_all:
            motion_name = motion_ids[sample_idx]
        else:
            motion_name = f'sample_{sample_idx}'

        all_motion_names.append(motion_name)
        print(f'\n### Processing sample {sample_idx + 1}/{args.num_samples} ({motion_name})')

        if cached_data is not None:
            input_motions, model_kwargs = data_source[sample_idx]
        else:
            input_motions, model_kwargs = next(data_source)
        input_motions = input_motions.to(dist_util.dev())
        texts = [args.text_condition] * args.batch_size
        model_kwargs['y']['text'] = texts
        if args.text_condition == '':
            args.guidance_param = 0.  # Force unconditioned generation

        # add inpainting mask according to args
        assert max_frames == input_motions.shape[-1]
        gt_frames_per_sample = {}
        model_kwargs['y']['inpainted_motion'] = input_motions
        if args.edit_mode == 'in_between':
            model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                                   device=input_motions.device)  # True means use gt motion
            for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
                start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
                gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
                model_kwargs['y']['inpainting_mask'][i, :, :,
                start_idx: end_idx] = False  # do inpainting in those frames
        elif args.edit_mode == 'upper_body':
            model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
                                                                device=input_motions.device)  # True is lower body data
            model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

        # Store input motion for visualization
        all_input_motions.append(input_motions.cpu())
        all_gt_frames.append(gt_frames_per_sample)

        for rep_i in range(args.num_repetitions):
            print(f'  Repetition {rep_i + 1}/{args.num_repetitions}')

            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_text += model_kwargs['y']['text']
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        print(f"  Total samples created: {len(all_motions) * args.batch_size}")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions,
             'motion_names': all_motion_names})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation for all input motions
    if model.data_rep == 'hml_vec':
        processed_input_motions = []
        for input_motion in all_input_motions:
            processed = data.dataset.t2m_dataset.inv_transform(input_motion.permute(0, 2, 3, 1)).float()
            processed = recover_from_ric(processed, n_joints)
            processed = processed.view(-1, *processed.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
            processed_input_motions.append(processed[0])  # Take first item since batch_size=1
        all_input_motions = processed_input_motions

    sample_print_template, row_print_template, all_print_template, \
    sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
    max_vis_samples = 6
    num_vis_samples = min(args.num_samples, max_vis_samples)
    animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
    max_length = max(all_lengths)

    for sample_i in range(args.num_samples):
        motion_name = all_motion_names[sample_i]

        # Generate input motion video (temporary)
        caption_input = 'Input Motion'
        length_input = all_lengths[sample_i * args.num_repetitions]
        motion_input = all_input_motions[sample_i].transpose(2, 0, 1)[:length_input]
        input_video_path = os.path.join(out_path, f'_temp_input_{sample_i}.mp4')
        gt_frames_for_sample = all_gt_frames[sample_i].get(0, [])

        ani_input = plot_3d_motion(input_video_path, skeleton, motion_input, title=caption_input,
                                    dataset=args.dataset, fps=fps, vis_mode='gt',
                                    gt_frames=gt_frames_for_sample)
        ani_input = ani_input.set_duration(length_input / fps)
        ani_input.write_videofile(input_video_path, fps=fps, codec='libx264', verbose=False, logger=None)

        # Generate output motion videos (temporary, one per repetition)
        rep_files = [input_video_path]
        for rep_i in range(args.num_repetitions):
            caption = all_text[sample_i * args.num_repetitions + rep_i]
            if caption == '':
                caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
            else:
                caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
            length = all_lengths[sample_i * args.num_repetitions + rep_i]
            motion = all_motions[sample_i * args.num_repetitions + rep_i].transpose(2, 0, 1)[:length]
            temp_output_path = os.path.join(out_path, f'_temp_output_{sample_i}_{rep_i}.mp4')
            gt_frames = all_gt_frames[sample_i].get(0, [])

            ani = plot_3d_motion(temp_output_path,
                                 skeleton, motion, dataset=args.dataset, title=caption,
                                 fps=fps, gt_frames=gt_frames)
            ani = ani.set_duration(length / fps)
            ani.write_videofile(temp_output_path, fps=fps, codec='libx264', verbose=False, logger=None)
            animations[sample_i, rep_i] = ani
            rep_files.append(temp_output_path)

        # Combine into side-by-side video with naming: sample_MOTIONID.mp4
        final_video_path = os.path.join(out_path, f'sample_{motion_name}.mp4')
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {final_video_path}'
        os.system(ffmpeg_rep_cmd)
        print(f'[{sample_i}] {motion_name} -> {final_video_path}')

        # Clean up temporary files
        for temp_file in rep_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        for rep_i in range(args.num_repetitions):
            temp_output = os.path.join(out_path, f'_temp_output_{sample_i}_{rep_i}.mp4')
            if os.path.exists(temp_output):
                os.remove(temp_output)

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
