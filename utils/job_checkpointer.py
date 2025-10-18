import copy
import os
import traceback

import torch
from .ddp_utils import print_rank0
from easydict import EasyDict as edict
from .optimizer_scheduler import configure_lr_scheduler
from rich import print


def checkpoint_job(
    out_dir,
    model,
    optimizer,
    lr_scheduler,
    fwdbwd_pass_step,
    param_update_step,
    s3_path=None,
):
    """Save the model and optimizer states."""
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module

    # Exclude the point encoder weights
    if hasattr(model, "point_encoder"):
        point_encoder = copy.deepcopy(model.point_encoder)
        model.point_encoder = None

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": fwdbwd_pass_step,
        "param_update_step": param_update_step,
    }

    os.makedirs(out_dir, exist_ok=True)
    ckpt_fpath = os.path.join(out_dir, f"ckpt_{fwdbwd_pass_step:016}.pt")
    torch.save(checkpoint, ckpt_fpath)
    print(f"Saved checkpoint to {os.path.abspath(ckpt_fpath)}")

    # Save to S3
    if s3_path is not None:
        s3_ckpt_path = os.path.join(s3_path, f"ckpt/ckpt_{fwdbwd_pass_step:016}.pt")
        print(f"Uploading {ckpt_fpath} to {s3_ckpt_path}")
        os.system(
            f"flash_s3_upload --local-dir {ckpt_fpath} --s3-url {s3_ckpt_path} --target-include-name"
        )

    # Restore the point encoder weights
    if hasattr(model, "point_encoder"):
        model.point_encoder = point_encoder


def checkpoint_job_s3(
    s3_path, model, optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step
):
    """Save the model and optimizer states."""
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": fwdbwd_pass_step,
        "param_update_step": param_update_step,
    }

    s3_ckpt_path = os.path.join(s3_path, f"ckpt_{fwdbwd_pass_step:016}.pt")
    async_upload = save_ckpt_to_s3(checkpoint, s3_ckpt_path)
    print(f"Saved checkpoint to {s3_ckpt_path}")


def find_checkpoints(out_dir):
    """Find the checkpoints in the output directory."""
    prefix, suffix = "ckpt_", ".pt"
    ckpt_names = [
        x for x in os.listdir(out_dir) if x.startswith(prefix) and x.endswith(suffix)
    ]
    ckpt_names = sorted(ckpt_names, key=lambda x: x[len(prefix) : -len(suffix)])
    ckpt_paths = [os.path.join(out_dir, ckpt_name) for ckpt_name in ckpt_names]

    return ckpt_paths


def resume_job(
    load_path,
    checkpoint_dir,
    model,
    optimizer,
    lr_scheduler,
    job_overview,
    warmup,
    reset_lr=False,
    reset_weight_decay=False,
    reset_training_state=False,
):
    """
    Resume training from the latest checkpoint in the output directory.
    Returns the fwdbwd_pass_step and param_update_step.

    Args:
        load_path: If dir, load the last checkpoint in the directory.
            O.w., assume it's a ckpt and load it.
        model: model to be loaded
        optimizer: optimizer to be loaded
        lr_scheduler:
        job_overview:
        warmup: warmup steps, only works if reset_lr is True.
        reset_lr: reset the training lr; Note that
        reset_weight_decay:

    Returns:
        optimizer, lr_scheduler, 0, 0

    """
    if load_path.startswith("s3://"):
        local_checkpoint_path = os.path.join(checkpoint_dir, "resume_ckpt.pt")
        print(f"Downloading checkpoints from {load_path} to {local_checkpoint_path}")
        os.system(
            f"flash_s3_download --s3-url {load_path} --local-dir {local_checkpoint_path} --target-include-name"
        )
        all_ckpt_paths = [local_checkpoint_path]
    else:
        if os.path.isdir(load_path):
            all_ckpt_paths = find_checkpoints(load_path)

            # No checkpoint found in this directory
            if len(all_ckpt_paths) == 0:
                return optimizer, lr_scheduler, 0, 0
        else:
            # If file, assume that it is a checkpoint
            if not load_path.endswith(".pt"):
                return optimizer, lr_scheduler, 0, 0

            all_ckpt_paths = [load_path]

    # Load the latest checkpoint in the reverse order
    #   This is to avoid the last checkpoint corrupted (due to disk issue or sudden kill jobs)
    for ckpt_fpath in all_ckpt_paths[::-1]:
        try:
            # Load checkpoints to CPU, it can avoid double loading the params into a single GPU.
            checkpoint = torch.load(ckpt_fpath, map_location="cpu")
        except:
            traceback.print_exc()
            print(
                f"Failed to load {ckpt_fpath}, we will continue to load the next ckpt in the reverse order"
            )
            continue
        else:
            break
    else:
        print(
            f"Failed to load any checkpoint in {load_path}; all ckpt paths: {all_ckpt_paths}"
        )
        return optimizer, lr_scheduler, 0, 0

    # Load model weights
    if model is not None:
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module

        status = model.load_state_dict(checkpoint["model"], strict=False)
        print_rank0(
            f"Loaded model from {os.path.abspath(ckpt_fpath)}, the status is {status}"
        )

    # reset the training state
    if reset_training_state:
        print_rank0(
            f"Reset the training state to have fresh optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step"
        )
        return optimizer, lr_scheduler, 0, 0

    try:
        if reset_lr:
            for ckpt_param_group, param_group in zip(
                checkpoint["optimizer"]["param_groups"], optimizer.param_groups
            ):
                ckpt_param_group["lr"] = param_group["lr"]
                ckpt_param_group["initial_lr"] = param_group["initial_lr"]
            print_rank0(f"Reset peak learning rate to {ckpt_param_group['initial_lr']}")
            print_rank0(
                f"Reset current learning rate to {ckpt_param_group['initial_lr']}"
            )

        if reset_weight_decay:
            for ckpt_param_group, param_group in zip(
                checkpoint["optimizer"]["param_groups"], optimizer.param_groups
            ):
                if ckpt_param_group["weight_decay"] > 0.0:
                    ckpt_param_group["weight_decay"] = param_group["weight_decay"]
            print_rank0(f"Reset weight_decay to {ckpt_param_group['weight_decay']}")

        print(
            f"checkpoint['optimizer']['param_groups'][0]['lr'] = {checkpoint['optimizer']['param_groups'][0]['lr']}"
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"optimizer.param_groups[0]['lr'] = {optimizer.param_groups[0]['lr']}")
        print_rank0(f"Loaded optimizer from {os.path.abspath(ckpt_fpath)}")
    except:
        traceback.print_exc()
        print("No worry! We will continue!")

    try:
        if reset_lr:
            total_steps = (
                job_overview.num_param_updates - checkpoint["param_update_step"]
            )
            lr_scheduler = configure_lr_scheduler(
                optimizer,
                total_steps,
                warmup,
                scheduler_type="cosine",
            )
            print_rank0(
                f"Reset learning rate scheduler; warmup: {warmup}, total steps: {total_steps}"
            )
        else:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print_rank0(
                f"Loaded learning rate scheduler from {os.path.abspath(ckpt_fpath)}"
            )
    except:
        traceback.print_exc()
        print("No worry! We will continue!")

    return (
        optimizer,
        lr_scheduler,
        checkpoint["fwdbwd_pass_step"],
        checkpoint["param_update_step"],
    )


def get_job_overview(
    num_gpus,
    num_epochs,
    num_train_samples,
    batch_size_per_gpu,
    gradient_accumulation_steps,
    max_fwdbwd_passes=int(1e10),
):
    """Compute the total number of training steps."""
    batch_size_per_fwdbwd_pass = batch_size_per_gpu * num_gpus
    num_fwdbwd_passes_per_epoch = max(
        1, int(num_train_samples / batch_size_per_fwdbwd_pass)
    )
    batch_size_per_param_update = (
        batch_size_per_fwdbwd_pass * gradient_accumulation_steps
    )
    num_param_updates_per_epoch = int(
        num_fwdbwd_passes_per_epoch / gradient_accumulation_steps
    )

    num_epochs = min(
        num_epochs, int(max_fwdbwd_passes / num_fwdbwd_passes_per_epoch) + 1
    )
    overview = edict(
        batch_size_per_fwdbwd_pass=batch_size_per_fwdbwd_pass,
        batch_size_per_param_update=batch_size_per_param_update,
        num_fwdbwd_passes_per_epoch=num_fwdbwd_passes_per_epoch,
        num_param_updates_per_epoch=num_param_updates_per_epoch,
        num_fwdbwd_passes=num_fwdbwd_passes_per_epoch * num_epochs,
        num_param_updates=num_param_updates_per_epoch * num_epochs,
        num_epochs=num_epochs,
    )
    return overview
