"""
FastVideoCombine — GPU-direct NVENC H.264 encoder for ComfyUI.

Requires:
  pip install PyNvVideoCodec

Encodes frames entirely on GPU via NVIDIA's hardware encoder.
Only the compressed bitstream touches CPU (for MP4 muxing).
FFmpeg is used solely for container muxing and audio, never for encoding.
"""

from __future__ import annotations

import datetime
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.utils import ProgressBar

# ═══════════════════════════════════════════════════════════════════════════
# PyNvVideoCodec — required, no fallback
# ═══════════════════════════════════════════════════════════════════════════
import PyNvVideoCodec as _nvc  # type: ignore[import-untyped]

# DLPack compatibility patch — PyNvVideoCodec may expect a different
# __dlpack__ signature than what PyTorch provides.
if hasattr(torch.Tensor, "__dlpack__"):
    _orig_dlpack = torch.Tensor.__dlpack__

    def _patched_dlpack(self, stream=None, *a, **kw):
        if stream is not None:
            return _orig_dlpack(self, stream=stream)
        return _orig_dlpack(self)

    torch.Tensor.__dlpack__ = _patched_dlpack

logger = logging.getLogger("FastVideoCombine")

ENCODE_ARGS = ("utf-8", "backslashreplace")
BIGMAX = 2**53 - 1
NV12_BATCH_SIZE = 32


# ═══════════════════════════════════════════════════════════════════════════
# ComfyUI type helpers
# ═══════════════════════════════════════════════════════════════════════════
class MultiInput(str):
    def __new__(cls, string: str, allowed_types: str | list = "*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types
        return res

    def __ne__(self, other: object) -> bool:
        if self.allowed_types == "*" or other == "*":
            return False
        return other not in self.allowed_types


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


class ContainsAll(dict):
    def __contains__(self, other: object) -> bool:
        return True

    def __getitem__(self, key: str):
        return super().get(key, (None, {}))


# ═══════════════════════════════════════════════════════════════════════════
# FFmpeg discovery (muxing only)
# ═══════════════════════════════════════════════════════════════════════════
def _ffmpeg_suitability(path: str) -> int:
    try:
        version = subprocess.run(
            [path, "-version"], check=True, capture_output=True,
        ).stdout.decode(*ENCODE_ARGS)
    except Exception:
        return 0
    score = 0
    for term, weight in [("libvpx", 20), ("264", 10), ("265", 3),
                         ("svtav1", 5), ("libopus", 1)]:
        if term in version:
            score += weight
    idx = version.find("2000-2")
    if idx >= 0:
        year = version[idx + 6 : idx + 9]
        if year.isnumeric():
            score += int(year)
    return score


def _find_ffmpeg() -> str | None:
    forced = os.environ.get("VHS_FORCE_FFMPEG_PATH")
    if forced:
        return forced
    candidates: list[str] = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore[import-untyped]
        candidates.append(get_ffmpeg_exe())
    except Exception:
        pass
    if os.environ.get("VHS_USE_IMAGEIO_FFMPEG") and candidates:
        return candidates[0]
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        candidates.append(sys_ff)
    for name in ("ffmpeg", "ffmpeg.exe"):
        if os.path.isfile(name):
            candidates.append(os.path.abspath(name))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=_ffmpeg_suitability)


ffmpeg_path: str | None = _find_ffmpeg()


# ═══════════════════════════════════════════════════════════════════════════
# Tensor helpers
# ═══════════════════════════════════════════════════════════════════════════
def _tensor_to_bytes(tensor: torch.Tensor) -> np.ndarray:
    """[H,W,C] float32 → uint8 numpy. Used only for first-frame PNG."""
    t = tensor.cpu().numpy() * 255 + 0.5
    return np.clip(t, 0, 255).astype(np.uint8)


def _to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]


# ═══════════════════════════════════════════════════════════════════════════
# GPU-direct NV12 conversion — BT.709 limited range
#
# Input:  [N, H, W, 3] float32 RGB in [0, 1] on CUDA
# Output: list of N × [H*3//2, W] uint8 NV12 on CUDA
#
# BT.709: Kr=0.2126, Kg=0.7152, Kb=0.0722
# Y  = 16  + 219·(Kr·R + Kg·G + Kb·B)
# Cb = 128 + 224·0.5·(B - Y')/(1 - Kb)
# Cr = 128 + 224·0.5·(R - Y')/(1 - Kr)
# ═══════════════════════════════════════════════════════════════════════════
def _rgb_batch_to_nv12(batch: torch.Tensor) -> list[torch.Tensor]:
    """Batched RGB→NV12 entirely on GPU. Returns list of contiguous NV12 frames."""
    N, H, W, _C = batch.shape
    R, G, B = batch[..., 0], batch[..., 1], batch[..., 2]

    Y = (16.0 + 46.559 * R + 156.629 * G + 15.812 * B).clamp_(16, 235)
    Cb = (128.0 - 25.656 * R - 86.344 * G + 112.0 * B).clamp_(16, 240)
    Cr = (128.0 + 112.0 * R - 101.684 * G - 10.316 * B).clamp_(16, 240)

    Y_u8 = Y.to(torch.uint8)
    Cb_sub = Cb.reshape(N, H // 2, 2, W // 2, 2).mean(dim=(2, 4)).to(torch.uint8)
    Cr_sub = Cr.reshape(N, H // 2, 2, W // 2, 2).mean(dim=(2, 4)).to(torch.uint8)
    UV = torch.stack([Cb_sub, Cr_sub], dim=-1).reshape(N, H // 2, W)
    nv12 = torch.cat([Y_u8, UV], dim=1)  # [N, H*3//2, W]
    return [nv12[i].contiguous() for i in range(N)]


# ═══════════════════════════════════════════════════════════════════════════
# Metadata
# ═══════════════════════════════════════════════════════════════════════════
def _prepare_metadata_file(video_metadata: dict) -> str | None:
    if not video_metadata:
        return None
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    path = os.path.join(folder_paths.get_temp_directory(), "fast_vc_metadata.txt")

    def esc(key: str, value) -> str:
        v = str(value)
        for ch, rpl in [("\\", "\\\\"), (";", "\\;"), ("#", "\\#"),
                        ("=", "\\="), ("\n", "\\\n")]:
            v = v.replace(ch, rpl)
        return f"{key}={v}"

    with open(path, "w") as f:
        f.write(";FFMETADATA1\n")
        if "prompt" in video_metadata:
            f.write(esc("prompt", json.dumps(video_metadata["prompt"])) + "\n")
        if "workflow" in video_metadata:
            f.write(esc("workflow", json.dumps(video_metadata["workflow"])) + "\n")
        for k, v in video_metadata.items():
            if k not in ("prompt", "workflow"):
                f.write(esc(k, json.dumps(v)) + "\n")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Encoder state — survives across meta_batch runs
# ═══════════════════════════════════════════════════════════════════════════
class _EncoderState:
    __slots__ = ("encoder", "raw_file", "raw_path", "total")

    def __init__(self, encoder, raw_file, raw_path: str):
        self.encoder = encoder
        self.raw_file = raw_file
        self.raw_path = raw_path
        self.total = 0

    def encode_batch(self, nv12_frames: list[torch.Tensor], pbar: ProgressBar):
        for nv12 in nv12_frames:
            bitstream = self.encoder.Encode(nv12)
            if len(bitstream) > 0:
                self.raw_file.write(bytearray(bitstream))
            self.total += 1
            pbar.update(1)

    def finalize(self) -> int:
        remaining = self.encoder.EndEncode()
        if len(remaining) > 0:
            self.raw_file.write(bytearray(remaining))
        self.raw_file.close()
        return self.total


def _create_encoder_state(
    file_path: str, width: int, height: int,
    frame_rate: float, bitrate_mbps: int,
) -> _EncoderState:
    raw_path = file_path + ".raw.h264"
    encoder = _nvc.CreateEncoder(
        width=width,
        height=height,
        fmt="NV12",
        usecpuinputbuffer=False,
        codec="h264",
        preset="P4",
        gpu_id=torch.cuda.current_device(),
        rc="vbr",
        bitrate=bitrate_mbps * 1_000_000,
        maxbitrate=bitrate_mbps * 2_000_000,
        fps=int(frame_rate),
        cudastream=torch.cuda.current_stream().cuda_stream,
    )
    raw_file = open(raw_path, "wb")
    return _EncoderState(encoder, raw_file, raw_path)


def _mux_to_mp4(
    raw_path: str, file_path: str, frame_rate: float,
    video_metadata: dict, save_metadata: bool,
) -> None:
    """Mux raw H.264 bitstream into MP4 container via FFmpeg."""
    if ffmpeg_path is None:
        raise ProcessLookupError(
            "ffmpeg is required for MP4 muxing and could not be found.\n"
            "Install imageio-ffmpeg, place ffmpeg in the working dir, or add to PATH."
        )
    mux_args = [
        ffmpeg_path, "-v", "error", "-y",
        "-r", str(frame_rate),
        "-f", "h264", "-i", raw_path,
        "-c:v", "copy",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
    ]
    if save_metadata and video_metadata:
        meta_path = _prepare_metadata_file(video_metadata)
        if meta_path:
            mux_args += [
                "-i", meta_path,
                "-map", "0:v", "-map_metadata", "1",
                "-movflags", "use_metadata_tags",
            ]
    mux_args.append(file_path)
    try:
        subprocess.run(mux_args, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("Muxing failed: %s", exc.stderr.decode(*ENCODE_ARGS))
        os.rename(raw_path, file_path)
        return
    try:
        os.remove(raw_path)
    except OSError:
        pass


def _encode_frames(images, enc_state: _EncoderState, pbar: ProgressBar) -> None:
    """Feed frames through the encoder in GPU-optimal batches."""
    batch_buf: list[torch.Tensor] = []

    def flush():
        if not batch_buf:
            return
        stacked = torch.stack(batch_buf)
        if not stacked.is_cuda:
            stacked = stacked.cuda()
        nv12_frames = _rgb_batch_to_nv12(stacked)
        enc_state.encode_batch(nv12_frames, pbar)
        batch_buf.clear()

    if isinstance(images, torch.Tensor):
        # Fast path: direct slicing, no list accumulation
        for i in range(0, len(images), NV12_BATCH_SIZE):
            chunk = images[i : i + NV12_BATCH_SIZE]
            if not chunk.is_cuda:
                chunk = chunk.cuda()
            nv12_frames = _rgb_batch_to_nv12(chunk)
            enc_state.encode_batch(nv12_frames, pbar)
    else:
        for tensor in images:
            batch_buf.append(tensor)
            if len(batch_buf) >= NV12_BATCH_SIZE:
                flush()
        flush()


# ═══════════════════════════════════════════════════════════════════════════
# The Node
# ═══════════════════════════════════════════════════════════════════════════
class FastVideoCombine:
    """GPU-direct NVENC H.264 encoder for ComfyUI.

    Encodes frames entirely on GPU via PyNvVideoCodec.
    FFmpeg is used only for MP4 muxing and audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 8, "min": 1, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "bitrate": ("INT", {"default": 40, "min": 1, "max": 999, "step": 1}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "save_metadata": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
            },
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }),
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "FastVideoCombine"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix: str = "AnimateDiff",
        bitrate: int = 40,
        pingpong: bool = False,
        save_output: bool = True,
        save_metadata: bool = True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        **kwargs,
    ):
        if latents is not None:
            images = latents
        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images["samples"]
            else:
                vae = None
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)

        num_frames = len(images)
        pbar = ProgressBar(num_frames)

        # ── VAE decode ───────────────────────────────────────────────────
        if vae is not None:
            downscale = getattr(vae, "downscale_ratio", 8)
            w, h = images.size(-1) * downscale, images.size(-2) * downscale
            fpb = (1920 * 1080 * 16) // (w * h) or 1

            def _batched_vae(imgs, _vae, n):
                it = iter(imgs)
                while batch := list(itertools.islice(it, n)):
                    yield from _vae.decode(torch.from_numpy(np.array(batch)))

            images = _batched_vae(images, vae, fpb)
            first_image = next(images)
            images = itertools.chain([first_image], images)
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]

        # ── output paths ─────────────────────────────────────────────────
        filename_prefix = datetime.datetime.now().strftime(filename_prefix)
        output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, _, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )
        output_files: list[str] = []

        # ── metadata ─────────────────────────────────────────────────────
        png_metadata = PngInfo()
        video_metadata: dict = {}
        if prompt is not None:
            png_metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                png_metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get("workflow", {}).get("extra", {})
        else:
            extra_options = {}
        png_metadata.add_text(
            "CreationTime", datetime.datetime.now().isoformat(" ")[:19],
        )

        # ── file counter ─────────────────────────────────────────────────
        if meta_batch is not None and unique_id in meta_batch.outputs:
            counter, enc_state = meta_batch.outputs[unique_id]
        else:
            max_counter = 0
            matcher = re.compile(
                rf"{re.escape(filename)}_(\d+)\D*\..+", re.IGNORECASE,
            )
            for existing in os.listdir(full_output_folder):
                m = matcher.fullmatch(existing)
                if m:
                    max_counter = max(max_counter, int(m.group(1)))
            counter = max_counter + 1
            enc_state = None

        # ── first-frame PNG (workflow drag-and-drop) ─────────────────────
        first_image_file = f"{filename}_{counter:05}.png"
        png_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get("VHS_MetadataImage", True) is not False:
            Image.fromarray(_tensor_to_bytes(first_image)).save(
                png_path, pnginfo=png_metadata, compress_level=4,
            )
        output_files.append(png_path)

        # ── dimension alignment (NVENC requires even dims) ───────────────
        dim_alignment = 2
        needs_pad = (
            first_image.shape[1] % dim_alignment
            or first_image.shape[0] % dim_alignment
        )
        if needs_pad:
            to_pad = (
                -first_image.shape[1] % dim_alignment,
                -first_image.shape[0] % dim_alignment,
            )
            padding = (
                to_pad[0] // 2, to_pad[0] - to_pad[0] // 2,
                to_pad[1] // 2, to_pad[1] - to_pad[1] // 2,
            )
            padfunc = torch.nn.ReplicationPad2d(padding)

            def _pad(image):
                return padfunc(
                    image.permute(2, 0, 1).to(dtype=torch.float32),
                ).permute(1, 2, 0)

            images = map(_pad, images)
            dimensions = (
                first_image.shape[1] + to_pad[0],
                first_image.shape[0] + to_pad[1],
            )
        else:
            dimensions = (first_image.shape[1], first_image.shape[0])

        # ── pingpong ─────────────────────────────────────────────────────
        if pingpong:
            images = _to_pingpong(images)
            if num_frames > 2:
                num_frames += num_frames - 2
                pbar.total = num_frames

        # ── loop (repeat frames at encode time) ──────────────────────────
        if loop_count > 0:
            images = list(images) if not isinstance(images, list) else images
            images = images * (loop_count + 1)
            num_frames = len(images)
            pbar.total = num_frames

        # ── output file path ─────────────────────────────────────────────
        file = f"{filename}_{counter:05}.mp4"
        file_path = os.path.join(full_output_folder, file)

        # ── NVENC encode ─────────────────────────────────────────────────
        if enc_state is None:
            enc_state = _create_encoder_state(
                file_path, dimensions[0], dimensions[1],
                frame_rate, bitrate,
            )
            if meta_batch is not None:
                meta_batch.outputs[unique_id] = (counter, enc_state)

        _encode_frames(images, enc_state, pbar)

        # ── meta_batch: defer finalize until all batches done ────────────
        if meta_batch is not None:
            try:
                from videohelpersuite.utils import requeue_workflow
            except ImportError:
                requeue_workflow = None
            if requeue_workflow is not None:
                requeue_workflow(
                    (meta_batch.unique_id, not meta_batch.has_closed_inputs),
                )
            if not meta_batch.has_closed_inputs:
                return {
                    "ui": {"unfinished_batch": [True]},
                    "result": ((save_output, []),),
                }
            # Final batch — finalize
            meta_batch.outputs.pop(unique_id, None)
            if len(meta_batch.outputs) == 0:
                meta_batch.reset()

        # ── finalize: flush encoder + mux to MP4 ─────────────────────────
        total_frames_output = enc_state.finalize()
        _mux_to_mp4(
            enc_state.raw_path, file_path, frame_rate,
            video_metadata, save_metadata,
        )
        output_files.append(file_path)

        # ── audio muxing ─────────────────────────────────────────────────
        a_waveform = None
        if audio is not None:
            try:
                a_waveform = audio["waveform"]
            except Exception:
                pass

        if a_waveform is not None and ffmpeg_path is not None:
            audio_file = f"{filename}_{counter:05}-audio.mp4"
            audio_path = os.path.join(full_output_folder, audio_file)

            ch = audio["waveform"].size(1)
            min_dur = total_frames_output / frame_rate + 1
            mux_args = [
                ffmpeg_path, "-v", "error", "-n",
                "-i", file_path,
                "-ar", str(audio["sample_rate"]),
                "-ac", str(ch),
                "-f", "f32le", "-i", "-",
                "-c:v", "copy",
                "-c:a", "aac",
                "-af", f"apad=whole_dur={min_dur}",
                "-shortest", audio_path,
            ]
            audio_data = (
                audio["waveform"].squeeze(0).transpose(0, 1).numpy().tobytes()
            )
            try:
                res = subprocess.run(
                    mux_args, input=audio_data, capture_output=True, check=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Audio mux failed:\n" + exc.stderr.decode(*ENCODE_ARGS),
                )
            if res.stderr:
                print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
            output_files.append(audio_path)
            file = audio_file

        # ── cleanup intermediates ────────────────────────────────────────
        if extra_options.get("VHS_KeepIntermediate", True) is False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)

        return {
            "ui": {
                "gifs": [
                    {
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output" if save_output else "temp",
                        "format": "video/h264-mp4",
                        "frame_rate": frame_rate,
                        "workflow": first_image_file,
                        "fullpath": output_files[-1],
                    }
                ],
            },
            "result": ((save_output, output_files),),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "FastVideoCombine": FastVideoCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVideoCombine": "Fast Video Combine \u26a1",
}
