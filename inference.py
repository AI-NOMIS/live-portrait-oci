# coding: utf-8
"""
for human
"""

import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    # source: assets/users/01_AbsoluteReality.png
    # driving: assets/users/905h7s1715682358236.mp4
    # output_dir: animations/
    # flag_use_half_precision: True
    # flag_crop_driving_video: False
    # device_id: 0
    # flag_force_cpu: False
    # flag_normalize_lip: True
    # flag_source_video_eye_retargeting: False
    # flag_video_editing_head_rotation: False
    # flag_eye_retargeting: False
    # flag_lip_retargeting: False
    # flag_stitching: True
    # flag_relative_motion: True
    # flag_pasteback: True
    # flag_do_crop: True
    # driving_option: expression-friendly
    # driving_multiplier: 1.0
    # driving_smooth_observation_variance: 3e-07
    # audio_priority: driving
    # det_thresh: 0.15
    # scale: 2.3
    # vx_ratio: 0
    # vy_ratio: -0.125
    # flag_do_rot: True
    # source_max_dim: 1280
    # source_division: 2
    # scale_crop_driving_video: 2.2
    # vx_ratio_crop_driving_video: 0.0
    # vy_ratio_crop_driving_video: -0.1
    # server_port: 8890
    # share: False
    # server_name: 127.0.0.1
    # flag_do_torch_compile: False
    # gradio_temp_dir: None

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )
    print(args)
    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    result = live_portrait_pipeline.execute(args)
    print(result)

class Predictor():
     def predict(source, driving):
        # set tyro theme
        tyro.extras.set_accent_color("bright_cyan")
        args = tyro.cli(ArgumentConfig)

        args.source = source
        args.driving = driving

        ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
        if osp.exists(ffmpeg_dir):
            os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )
        print(args)
        fast_check_args(args)

        # specify configs for inference
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)

        live_portrait_pipeline = LivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )

        # run
        result = live_portrait_pipeline.execute(args)
        return result


if __name__ == "__main__":
    Predictor.predict()
