#!/usr/bin/env python3

"""Utility script to pretrain DINOv3 backbones on an unlabeled image folder."""

"""Utility script to pretrain ViT-B on an unlabeled image folder using DINOv3."""


from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import List

import torch
from dinov3.configs import setup_config, setup_job
from dinov3.logging import setup_logging
from dinov3.train import train as train_module
from dinov3.train.train import (
    MultiDistillationMetaArch,
    SSLMetaArch,
    do_test,
    do_train,
)

LOGGER = logging.getLogger("dinov3")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "dinov3" / "configs" / "train" / "multidist_tests" / "vitb_p16.yaml"

DEFAULT_OUTPUT_DIRS = {
    "vit_base": Path("./outputs/vitb-folder").resolve(),
    "convnext_small": Path("./outputs/convnexts-folder").resolve(),
}
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIRS["vit_base"]
CONVNEXT_SMALL_DEFAULT_WEIGHTS = "dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth"

DEFAULT_OUTPUT_DIR = Path("./outputs/vitb-folder").resolve()



def build_parser() -> argparse.ArgumentParser:
    base_parser = train_module.get_args_parser(add_help=False)
    parser = argparse.ArgumentParser(

        description="Self-supervised DINOv3 training on a directory of unlabeled images.",

        description="Self-supervised ViT-B training on a directory of unlabeled images.",

        parents=[base_parser],
        add_help=True,
    )
    parser.add_argument(

        "--arch",
        default="vit_base",
        help="Backbone architecture to train (e.g. vit_base, convnext_small).",
    )


        "--images-path",
        default=str(Path("./data/unlabeled_images")),
        help="Path to the folder that contains training images.",
    )
    parser.add_argument(
        "--extensions",
        default="",
        help="Optional list of image extensions (comma separated) to keep. Defaults to common image types.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Disable recursive directory traversal.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-GPU batch size. Defaults to 16 for single GPU training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Base learning rate for AdamW.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=0,
        help="Freeze the first N transformer blocks of the student backbone.",
    )
    parser.add_argument(
        "--freeze-patch-embed",
        action="store_true",
        help="Freeze the patch embedding and positional parameters of the student backbone.",
    )
    parser.add_argument(
        "--resume-teacher",
        default="",
        help="Optional path to a teacher checkpoint to warm-start the student.",
    )

    parser.add_argument(
        "--pretrained-weights",
        default="",
        help=(
            "Optional pretrained ViT checkpoint (local path, torch.hub identifier, or URL). "
            "Weights are matched against the student backbone automatically."
        ),
    )


    parser.add_argument(
        "--pretrained-weights",
        default="",
        help=(
            "Optional path to a pretrained ViT checkpoint. "
            "Weights are matched against the student backbone automatically."
        ),
    )



    parser.set_defaults(
        config_file=str(DEFAULT_CONFIG),
        output_dir=str(DEFAULT_OUTPUT_DIR),
    )
    return parser


def _build_dataset_option(args: argparse.Namespace) -> str:
    image_root = Path(args.images_path).expanduser().resolve()
    if not image_root.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_root}")
    tokens: List[str] = ["UnlabeledImageFolder", f"root={image_root}"]
    if not args.recursive:
        tokens.append("recursive=False")
    extensions = [ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()]
    if extensions:
        tokens.append(f"extensions={'|'.join(extensions)}")
    return ":".join(tokens)


def _prepare_opts(args: argparse.Namespace) -> None:
    args.opts = list(args.opts or [])
    args.opts.append(f"train.dataset_path={_build_dataset_option(args)}")

    args.opts.append(f"student.arch={args.arch}")

    args.opts.append("student.arch=vit_base")

    args.opts.append(f"train.batch_size_per_gpu={args.batch_size}")
    args.opts.append(f"train.num_workers={args.num_workers}")
    args.opts.append(f"optim.epochs={args.epochs}")
    args.opts.append(f"optim.lr={args.learning_rate}")
    if args.freeze_blocks:
        args.opts.append(f"student.freeze_blocks={args.freeze_blocks}")
    if args.freeze_patch_embed:
        args.opts.append("student.freeze_patch_embed=True")
    if args.resume_teacher:
        teacher_path = Path(args.resume_teacher).expanduser().resolve()
        args.opts.append(f"student.resume_from_teacher_chkpt={teacher_path}")

    weights_spec = args.pretrained_weights.strip()
    if not weights_spec and args.arch == "convnext_small":
        candidate_locations = [
            Path(CONVNEXT_SMALL_DEFAULT_WEIGHTS),
            Path.cwd() / CONVNEXT_SMALL_DEFAULT_WEIGHTS,
            Path(__file__).resolve().parent / CONVNEXT_SMALL_DEFAULT_WEIGHTS,
            Path(__file__).resolve().parents[1] / CONVNEXT_SMALL_DEFAULT_WEIGHTS,
        ]
        for candidate in candidate_locations:
            candidate = candidate.expanduser()
            if candidate.exists():
                weights_spec = str(candidate.resolve())
                LOGGER.info("Using default ConvNeXt-S weights located at %s", weights_spec)
                break
        else:
            LOGGER.warning(
                "Default ConvNeXt-S weights '%s' were not found; proceeding without pretrained initialization.",
                CONVNEXT_SMALL_DEFAULT_WEIGHTS,
            )
    if weights_spec:
        candidate_path = Path(weights_spec).expanduser()
        if candidate_path.exists():
            weights_spec = str(candidate_path.resolve())
        else:
            LOGGER.info("Using non-local pretrained weights spec: %s", weights_spec)
        args.opts.append(f"student.pretrained_weights={weights_spec}")


    if args.pretrained_weights:
        weights_spec = args.pretrained_weights.strip()
        if weights_spec:
            candidate_path = Path(weights_spec).expanduser()
            if candidate_path.exists():
                weights_spec = str(candidate_path.resolve())
            else:
                LOGGER.info(
                    "Using non-local pretrained weights spec: %s", weights_spec
                )
            args.opts.append(f"student.pretrained_weights={weights_spec}")

    if args.pretrained_weights:
        weights_path = Path(args.pretrained_weights).expanduser().resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
        args.opts.append(f"student.pretrained_weights={weights_path}")



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.multi_distillation:
        raise RuntimeError("This helper script does not support multi-distillation runs.")


    if args.output_dir == str(DEFAULT_OUTPUT_DIR):
        arch_default = DEFAULT_OUTPUT_DIRS.get(args.arch)
        if arch_default is not None:
            args.output_dir = str(arch_default)


    _prepare_opts(args)

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        args.opts.append("MODEL.DEVICE=cpu")
        args.opts.append("compute_precision.param_dtype=fp32")
    setup_job(
        output_dir=args.output_dir,
        seed=args.seed,
        distributed_enabled=use_cuda,
    )
    if use_cuda:
        cfg = setup_config(args, strict_cfg=False)
    else:
        from dinov3.configs.config import get_cfg_from_args, write_config

        cfg = get_cfg_from_args(args, strict=False)
        if args.output_dir is not None:
            write_config(cfg, args.output_dir)
    LOGGER.info("%s", cfg)
    setup_logging(
        output=os.path.join(os.path.abspath(args.output_dir), "nan_logs"),
        name="nan_logger",
    )

    meta_arch = {
        "SSLMetaArch": SSLMetaArch,
        "MultiDistillationMetaArch": MultiDistillationMetaArch,
    }.get(cfg.MODEL.META_ARCHITECTURE, None)
    if meta_arch is None:
        raise ValueError(f"Unknown MODEL.META_ARCHITECTURE {cfg.MODEL.META_ARCHITECTURE}")
    LOGGER.info("Making meta arch %s", meta_arch.__name__)
    with torch.device("meta"):
        model = meta_arch(cfg)
    if use_cuda:
        model.prepare_for_distributed_training()
    target_device = torch.device("cuda" if use_cuda else "cpu")
    model._apply(
        lambda t: torch.full_like(
            t,
            fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
            device=target_device,
        ),
        recurse=True,
    )
    LOGGER.info("Model after distributed:\n%s", model)

    if args.eval_only:
        model.init_weights()
        iteration = (
            model.get_checkpointer_class()(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        do_test(cfg, model, f"manual_{iteration}")
        return

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    main()
