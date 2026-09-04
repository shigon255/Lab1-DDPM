import argparse
import numpy as np
import torch
from pathlib import Path
from dataset import save_traj_strip, tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDPMScheduler

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    # 1) load model
    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval().to(device)

    # 2) predictor / scheduler
    #    The scheduler stored in the checkpoint is reused, so sampling always
    #    runs with the exact beta schedule the model was trained with. --mode is
    #    only checked for consistency: silently sampling a cosine-trained model
    #    with a linear schedule yields pure noise and is very hard to debug.
    ddpm.var_scheduler = ddpm.var_scheduler.to(device)

    ckpt_predictor = ddpm.predictor  # None for checkpoints saved before it was recorded
    if args.predictor is None:
        if ckpt_predictor is None:
            raise ValueError(
                "This checkpoint does not record its predictor; pass --predictor."
            )
        ddpm.predictor = ckpt_predictor
    else:
        if ckpt_predictor is not None and args.predictor != ckpt_predictor:
            raise ValueError(
                f"--predictor {args.predictor} does not match the checkpoint "
                f"({ckpt_predictor}). Omit --predictor to use the checkpoint's."
            )
        ddpm.predictor = args.predictor

    ckpt_mode = getattr(ddpm.var_scheduler, "schedule_mode", None)
    if ckpt_mode is None:
        # Older checkpoints do not record their schedule. Sampling one with the
        # wrong beta schedule yields pure noise and no error, so ask rather than
        # guess.
        raise ValueError(
            "This checkpoint does not record its beta schedule; retrain with the "
            "current code, or rebuild the scheduler yourself before sampling."
        )
    if args.mode is not None and args.mode != ckpt_mode:
        raise ValueError(
            f"--mode {args.mode} does not match the checkpoint's beta schedule "
            f"({ckpt_mode}). Omit --mode to use the checkpoint's schedule."
        )
    print(f"Sampling with beta schedule '{ckpt_mode}' and predictor '{ddpm.predictor}'.")

    if args.save_traj:
        # One reverse trajectory, saved as a strip -- the figure report item 2 asks
        # for. Done first so it is available even if sampling is interrupted, and
        # written next to save_dir rather than inside it: measure_fid.py picks up
        # every .png under the directory it is given, and the strip is not a sample.
        traj_path = save_dir.parent / f"{save_dir.name}_traj.png"
        traj = ddpm.sample(1, return_traj=True)
        save_traj_strip(traj_path, traj, num_frames=args.traj_frames)
        print(f"Saved the trajectory strip to {traj_path}.")

    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:
            assert getattr(ddpm.network, "use_cfg", False), "This checkpoint wasn't trained with CFG."

            samples = ddpm.sample(
                B,
                class_label=torch.randint(0, 3, (B,), device=device),
                guidance_scale=args.cfg_scale,
            )
        else:
            samples = ddpm.sample(B)

        for j, img in zip(range(sidx, eidx), tensor_to_pil_image(samples)):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="number of images to generate (500 is what FID is measured on).")
    parser.add_argument("--save_traj", action="store_true",
                        help="also save one reverse-process trajectory as traj.png.")
    parser.add_argument("--traj_frames", type=int, default=10,
                        help="number of frames in the trajectory strip (at least 2).")

    parser.add_argument("--predictor", type=str, default=None,
                        choices=["noise", "x0", "mean"],
                        help="optional if the checkpoint records it; checked for consistency.")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["linear", "cosine", "quad"],
                        help="optional; only checked against the checkpoint's schedule.")


    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    if args.traj_frames < 2:
        parser.error("--traj_frames must be at least 2")
    main(args)
