"""
FastVideoCombine — Drop-in replacement for VideoHelperSuite's Video Combine node.

Optimised encoding pipeline:
  Tier 1  Batched GPU→CPU transfer + threaded ffmpeg pipe writes   (no extra deps)
  Tier 3  GPU-direct NVENC via VPF / PyNvVideoCodec                (optional)

Auto-selects the fastest available backend; falls back gracefully.
"""

from __future__ import annotations

import datetime
import functools
import itertools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from queue import Queue
from string import Template
from typing import List

import numpy as np
import torch
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.utils import ProgressBar

# ---------------------------------------------------------------------------
# Optional GPU-direct encoder (Tier 3)
# ---------------------------------------------------------------------------
_gpu_encoder_name: str | None = None

try:
    import PyNvCodec as nvc        # type: ignore[import-untyped]
    import PytorchNvCodec as pnvc  # type: ignore[import-untyped]
    _gpu_encoder_name = "VPF"
except ImportError:
    pass

if _gpu_encoder_name is None:
    try:
        import PyNvVideoCodec as pnvc_v2  # type: ignore[import-untyped]
        _gpu_encoder_name = "PyNvVideoCodec"
    except ImportError:
        pass

logger = logging.getLogger("FastVideoCombine")

# ═══════════════════════════════════════════════════════════════════════════
# Constants & tiny helpers
# ═══════════════════════════════════════════════════════════════════════════
ENCODE_ARGS = ("utf-8", "backslashreplace")
BIGMAX = 2**53 - 1


class MultiInput(str):
    """Allows a widget to accept multiple ComfyUI types."""
    def __new__(cls, string: str, allowed_types: str | list = "*"):
        res = super().__new__(cls, string)
        res.allowed_types = allowed_types  # type: ignore[attr-defined]
        return res

    def __ne__(self, other: object) -> bool:
        if self.allowed_types == "*" or other == "*":  # type: ignore[attr-defined]
            return False
        return other not in self.allowed_types  # type: ignore[attr-defined]


imageOrLatent = MultiInput("IMAGE", ["IMAGE", "LATENT"])
floatOrInt = MultiInput("FLOAT", ["FLOAT", "INT"])


class ContainsAll(dict):
    """Dict that claims to contain every key (for hidden inputs)."""
    def __contains__(self, other: object) -> bool:
        return True

    def __getitem__(self, key: str):
        return super().get(key, (None, {}))


def flatten_list(lst: list) -> list:
    out: list = []
    for e in lst:
        if isinstance(e, list):
            out.extend(e)
        else:
            out.append(e)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FFmpeg discovery  (adapted from VHS)
# ═══════════════════════════════════════════════════════════════════════════
def _ffmpeg_suitability(path: str) -> int:
    try:
        version = subprocess.run(
            [path, "-version"], check=True, capture_output=True
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
        year = version[idx + 6: idx + 9]
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
gifski_path: str | None = (
    os.environ.get("VHS_GIFSKI")
    or os.environ.get("JOV_GIFSKI")
    or shutil.which("gifski")
)


# ═══════════════════════════════════════════════════════════════════════════
# Video format loading  (adapted from VHS)
# ═══════════════════════════════════════════════════════════════════════════
_FORMATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_formats")

# Register with ComfyUI's folder system
if "VHS_video_formats" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ((), {".json"})
if len(folder_paths.folder_names_and_paths["VHS_video_formats"][1]) == 0:
    folder_paths.folder_names_and_paths["VHS_video_formats"][1].add(".json")


def iterate_format(video_format: dict, for_widgets: bool = True):
    """Yield widget definitions (for_widgets=True) or resolve arguments."""
    def indirector(cont, index):
        if isinstance(cont[index], list) and (
            not for_widgets
            or len(cont[index]) > 1
            and not isinstance(cont[index][1], dict)
        ):
            inp = yield cont[index]
            if inp is not None:
                cont[index] = inp
                yield
    for k in video_format:
        if k == "extra_widgets":
            if for_widgets:
                yield from video_format["extra_widgets"]
        elif k.endswith("_pass"):
            for i in range(len(video_format[k])):
                yield from indirector(video_format[k], i)
            if not for_widgets:
                video_format[k] = flatten_list(video_format[k])
        else:
            yield from indirector(video_format, k)


def _cached_formats(duration: float = 5.0):
    cached_ret = None
    cache_time = 0.0

    def get():
        nonlocal cached_ret, cache_time
        if time.time() > cache_time + duration or cached_ret is None:
            cache_time = time.time()
            cached_ret = _load_video_formats()
        return cached_ret
    return get


def _load_video_formats():
    format_files: dict[str, str] = {}
    try:
        for name in folder_paths.get_filename_list("VHS_video_formats"):
            format_files[name] = folder_paths.get_full_path("VHS_video_formats", name)
    except Exception:
        pass  # VHS not installed — that's fine
    for item in os.scandir(_FORMATS_DIR):
        if item.is_file() and item.name.endswith(".json"):
            format_files[item.name[:-5]] = item.path
    formats: list[str] = []
    widgets: dict[str, list] = {}
    for name, path in format_files.items():
        with open(path, "r") as f:
            vfmt = json.load(f)
        if "gifski_pass" in vfmt and gifski_path is None:
            continue
        ws = list(iterate_format(vfmt))
        formats.append("video/" + name)
        if ws:
            widgets["video/" + name] = ws
    return formats, widgets


get_video_formats = _cached_formats()


def apply_format_widgets(format_name: str, kwargs: dict) -> dict:
    local = os.path.join(_FORMATS_DIR, format_name + ".json")
    if os.path.exists(local):
        path = local
    else:
        path = folder_paths.get_full_path("VHS_video_formats", format_name)
    with open(path, "r") as f:
        video_format = json.load(f)
    for w in iterate_format(video_format):
        if w[0] not in kwargs:
            if len(w) > 2 and "default" in w[2]:
                default = w[2]["default"]
            else:
                if isinstance(w[1], list):
                    default = w[1][0]
                else:
                    default = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[w[1]]
            kwargs[w[0]] = default
    wit = iterate_format(video_format, False)
    for w in wit:
        while isinstance(w, list):
            if len(w) == 1:
                w = [Template(x).substitute(**kwargs) for x in w[0]]
                break
            elif isinstance(w[1], dict):
                w = w[1][str(kwargs[w[0]])]
            elif len(w) > 3:
                w = Template(w[3]).substitute(val=kwargs[w[0]])
            else:
                w = str(kwargs[w[0]])
        wit.send(w)
    return video_format


def merge_filter_args(args: list, ftype: str = "-vf") -> None:
    """Collapse duplicate -vf/-af into a single comma-separated filter chain."""
    try:
        start = args.index(ftype) + 1
        idx = start
        while True:
            idx = args.index(ftype, idx)
            args[start] += "," + args[idx + 1]
            args.pop(idx)
            args.pop(idx)
    except ValueError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Tensor helpers
# ═══════════════════════════════════════════════════════════════════════════
def tensor_to_int(tensor: torch.Tensor, bits: int) -> np.ndarray:
    t = tensor.cpu().numpy() * (2**bits - 1) + 0.5
    return np.clip(t, 0, 2**bits - 1)


def tensor_to_bytes(tensor: torch.Tensor) -> np.ndarray:
    return tensor_to_int(tensor, 8).astype(np.uint8)


def tensor_to_shorts(tensor: torch.Tensor) -> np.ndarray:
    return tensor_to_int(tensor, 16).astype(np.uint16)


def to_pingpong(inp):
    if not hasattr(inp, "__getitem__"):
        inp = list(inp)
    yield from inp
    for i in range(len(inp) - 2, 0, -1):
        yield inp[i]


# ═══════════════════════════════════════════════════════════════════════════
# Encoding backends
# ═══════════════════════════════════════════════════════════════════════════
def _batched_iter(it, n: int):
    """Yield batches of up to *n* items.

    If *it* is a torch.Tensor, yields zero-copy slices instead of stacking.
    """
    if isinstance(it, torch.Tensor):
        for i in range(0, len(it), n):
            yield it[i:i + n]
    else:
        while True:
            batch = list(itertools.islice(it, n))
            if not batch:
                return
            yield batch


def _optimal_batch_size(height: int, width: int, channels: int = 3,
                        max_vram_mb: int = 1536) -> int:
    """Pick a batch size that fits in *max_vram_mb* of VRAM head-room.

    During conversion we briefly hold both float32 (src) and uint8 (dst)
    copies of each batch on the GPU.
    """
    per_frame = height * width * channels * 5  # 4 (f32) + 1 (u8)
    n = max(1, int(max_vram_mb * 1024 * 1024 / per_frame))
    return min(n, 64)


# --- Tier 1: optimised ffmpeg subprocess --------------------------------

def _prepare_metadata_file(video_metadata: dict) -> str | None:
    """Write an ffmetadata1 file and return its path."""
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


def encode_ffmpeg_batched(
    args: list[str],
    video_format: dict,
    video_metadata: dict,
    file_path: str,
    env: dict,
    images_iter,
    pbar: ProgressBar,
    batch_size: int,
) -> int:
    """Tier-1 encoder: batched GPU→CPU + threaded pipe writer.

    Returns total frames written.
    """
    # ---- build final command -------------------------------------------
    metadata_path: str | None = None
    if video_format.get("save_metadata", "False") != "False":
        metadata_path = _prepare_metadata_file(video_metadata)

    if metadata_path:
        final_args = (
            args[:1]
            + ["-i", metadata_path]
            + args[1:]
            + ["-metadata", "creation_time=now",
               "-movflags", "use_metadata_tags",
               file_path]
        )
    else:
        final_args = args + [file_path]

    # ---- threaded pipe writer ------------------------------------------
    write_q: Queue[bytes | None] = Queue(maxsize=2)
    write_err: list[Exception | None] = [None]

    def _writer(proc: subprocess.Popen) -> None:
        try:
            while True:
                chunk = write_q.get()
                if chunk is None:
                    break
                proc.stdin.write(chunk)  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as exc:
            write_err[0] = exc
            # Drain remaining items so the main thread never blocks on put()
            while not write_q.empty():
                try:
                    write_q.get_nowait()
                except Exception:
                    break

    total: int = 0
    stderr_bytes = b""
    with subprocess.Popen(
        final_args,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        env=env,
    ) as proc:
        writer = threading.Thread(target=_writer, args=(proc,), daemon=True)
        writer.start()

        try:
            for batch in _batched_iter(images_iter, batch_size):
                if write_err[0] is not None:
                    break

                if isinstance(batch, torch.Tensor):
                    stacked = batch.clone()
                else:
                    stacked = torch.stack(batch)
                # In-place GPU conversion: 384MB uint8 output vs 4.5GB intermediates
                stacked.mul_(255).add_(0.5).clamp_(0, 255)
                raw = stacked.to(torch.uint8).cpu().numpy().tobytes()

                while write_err[0] is None:
                    try:
                        write_q.put(raw, timeout=2.0)
                        break
                    except Exception:
                        continue

                n = len(batch)
                total += n
                pbar.update(n)

        finally:
            try:
                write_q.put_nowait(None)
            except Exception:
                pass
            writer.join(timeout=30)

            if writer.is_alive():
                proc.kill()
                writer.join(timeout=5)

            if write_err[0] is not None:
                stderr_out = proc.stderr.read().decode(*ENCODE_ARGS) if proc.stderr else ""  # type: ignore[union-attr]
                raise RuntimeError(
                    "ffmpeg encoding failed:\n" + stderr_out
                ) from write_err[0]

            try:
                proc.stdin.flush()  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
            except BrokenPipeError:
                pass
            stderr_bytes = proc.stderr.read() if proc.stderr else b""  # type: ignore[union-attr]

    if stderr_bytes:
        logger.debug("ffmpeg stderr: %s", stderr_bytes.decode(*ENCODE_ARGS))

    return total


# --- legacy / fallback for pre_pass & gifski formats --------------------

def ffmpeg_process_legacy(args, video_format, video_metadata, file_path, env):
    """Generator-based encoder — used only for pre_pass / gifski formats
    that require the original VHS frame-by-frame approach."""
    res = None
    frame_data = yield
    total_frames_output = 0

    if video_format.get("save_metadata", "False") != "False":
        metadata_path = _prepare_metadata_file(video_metadata)
        if metadata_path:
            m_args = (
                args[:1]
                + ["-i", metadata_path]
                + args[1:]
                + ["-metadata", "creation_time=now",
                   "-movflags", "use_metadata_tags"]
            )
        else:
            m_args = None
    else:
        m_args = None

    proc_args = (m_args or args) + [file_path]
    with subprocess.Popen(
        proc_args, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=env
    ) as proc:
        try:
            while frame_data is not None:
                proc.stdin.write(frame_data)  # type: ignore[union-attr]
                frame_data = yield
                total_frames_output += 1
            proc.stdin.flush()  # type: ignore[union-attr]
            proc.stdin.close()  # type: ignore[union-attr]
            res = proc.stderr.read()  # type: ignore[union-attr]
        except BrokenPipeError:
            err = proc.stderr.read()  # type: ignore[union-attr]
            if os.path.exists(file_path):
                raise RuntimeError(
                    "ffmpeg error:\n" + err.decode(*ENCODE_ARGS)
                )
            logger.warning("Metadata save failed: %s", err.decode(*ENCODE_ARGS))

    if res is not None and res != b"":
        # retry without metadata
        with subprocess.Popen(
            args + [file_path], stderr=subprocess.PIPE,
            stdin=subprocess.PIPE, env=env
        ) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)  # type: ignore[union-attr]
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()  # type: ignore[union-attr]
                proc.stdin.close()  # type: ignore[union-attr]
                res = proc.stderr.read()  # type: ignore[union-attr]
            except BrokenPipeError:
                res = proc.stderr.read()  # type: ignore[union-attr]
                raise RuntimeError(
                    "ffmpeg error:\n" + res.decode(*ENCODE_ARGS)
                )

    yield total_frames_output
    if res and len(res) > 0:
        print(res.decode(*ENCODE_ARGS), end="", file=sys.stderr)


def gifski_process(args, dimensions, frame_rate, video_format, file_path, env):
    """Pass-through for gifski formats (unchanged from VHS)."""
    frame_data = yield
    with subprocess.Popen(
        args + video_format["main_pass"] + ["-f", "yuv4mpegpipe", "-"],
        stderr=subprocess.PIPE, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, env=env,
    ) as procff:
        with subprocess.Popen(
            [gifski_path]
            + video_format["gifski_pass"]
            + ["-W", str(dimensions[0]), "-H", str(dimensions[1])]
            + ["-r", str(frame_rate)]
            + ["-q", "-o", file_path, "-"],
            stderr=subprocess.PIPE, stdin=procff.stdout,
            stdout=subprocess.PIPE,
        ) as procgs:
            total = 0
            try:
                while frame_data is not None:
                    procff.stdin.write(frame_data)  # type: ignore[union-attr]
                    frame_data = yield
                    total += 1
                procff.stdin.flush()  # type: ignore[union-attr]
                procff.stdin.close()  # type: ignore[union-attr]
            except BrokenPipeError:
                pass
            procgs.wait()
    yield total


# --- CUDA hwaccel colorspace conversion (bypasses CPU swscale) ----------

_cuda_hwaccel_supported: bool | None = None


def _check_cuda_hwaccel() -> bool:
    """One-time probe: does this ffmpeg build have hwupload_cuda + scale_cuda?"""
    global _cuda_hwaccel_supported
    if _cuda_hwaccel_supported is not None:
        return _cuda_hwaccel_supported
    if ffmpeg_path is None:
        _cuda_hwaccel_supported = False
        return False
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=5,
        )
        has_both = "hwupload_cuda" in result.stdout and "scale_cuda" in result.stdout
        _cuda_hwaccel_supported = has_both
        if has_both:
            logger.info("CUDA hwaccel filters available — GPU colorspace conversion enabled")
    except Exception:
        _cuda_hwaccel_supported = False
    return _cuda_hwaccel_supported


def _inject_cuda_colorspace(args: list[str]) -> list[str]:
    """For NVENC codecs: replace CPU swscale with GPU-accelerated scale_cuda.

    Moves RGB→NV12 conversion from CPU (5-15ms/frame) to GPU (<1ms/frame).
    Returns modified args, or original args if not applicable.
    """
    if not _check_cuda_hwaccel():
        return args

    try:
        cv_idx = args.index("-c:v")
        codec = args[cv_idx + 1]
    except (ValueError, IndexError):
        return args
    if not isinstance(codec, str) or "nvenc" not in codec:
        return args

    args = list(args)

    for i in range(cv_idx, len(args) - 1):
        if args[i] == "-pix_fmt":
            args.pop(i)
            args.pop(i)
            break

    try:
        vf_idx = args.index("-vf")
        existing = args[vf_idx + 1]
        args[vf_idx + 1] = f"hwupload_cuda,scale_cuda=format=nv12,{existing}"
    except ValueError:
        args += ["-vf", "hwupload_cuda,scale_cuda=format=nv12"]

    return args


# --- Tier 3: GPU-direct NVENC via VPF -----------------------------------

def _is_nvenc_format(video_format: dict) -> bool:
    """Check if the format JSON uses an NVENC codec."""
    main_pass = video_format.get("main_pass", [])
    for i, arg in enumerate(main_pass):
        if arg == "-c:v" and i + 1 < len(main_pass):
            codec = main_pass[i + 1]
            if isinstance(codec, str) and "nvenc" in codec:
                return True
    return False


def _nvenc_codec_name(video_format: dict) -> str:
    """Extract the codec short-name (h264 / hevc / av1) from an nvenc format."""
    main_pass = video_format.get("main_pass", [])
    for i, arg in enumerate(main_pass):
        if arg == "-c:v" and i + 1 < len(main_pass):
            codec = main_pass[i + 1]
            if isinstance(codec, str):
                if "h264" in codec:
                    return "h264"
                if "hevc" in codec:
                    return "hevc"
                if "av1" in codec:
                    return "av1"
    return "h264"


def encode_gpu_direct(
    images_iter,
    file_path: str,
    width: int,
    height: int,
    frame_rate: float,
    video_format: dict,
    video_metadata: dict,
    pbar: ProgressBar,
    env: dict,
    bitrate_arg: list[str],
) -> int | None:
    """Tier-3: encode frames entirely on the GPU via VPF.

    Returns the number of frames encoded, or ``None`` if GPU encoding
    is unavailable / unsupported for this format, signalling the caller
    to fall back to Tier 1.
    """
    if _gpu_encoder_name != "VPF":
        return None
    if not _is_nvenc_format(video_format):
        return None

    codec = _nvenc_codec_name(video_format)
    gpu_id = 0

    bitrate_val = "10M"
    for i, a in enumerate(bitrate_arg):
        if a == "-b:v" and i + 1 < len(bitrate_arg):
            bitrate_val = bitrate_arg[i + 1]

    try:
        enc_params = {
            "preset": "P4",
            "codec": codec,
            "s": f"{width}x{height}",
            "bitrate": bitrate_val.replace("M", "000000").replace("K", "000"),
        }

        encoder = nvc.PyNvEncoder(enc_params, gpu_id)  # type: ignore[name-defined]

        to_nv12 = nvc.PySurfaceConverter(  # type: ignore[name-defined]
            width, height,
            nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.NV12,  # type: ignore[name-defined]
            gpu_id,
        )
        cc_ctx = nvc.ColorspaceConversionContext(  # type: ignore[name-defined]
            nvc.ColorSpace.BT_709, nvc.ColorRange.MPEG,  # type: ignore[name-defined]
        )
    except Exception as exc:
        logger.info("VPF encoder init failed (%s) — falling back to ffmpeg", exc)
        return None

    extension = video_format.get("extension", "mp4")
    raw_path = file_path + f".raw.{codec}"
    enc_frame = np.ndarray(shape=(0,), dtype=np.uint8)
    total = 0
    first_frame = True

    try:
        with open(raw_path, "wb") as f:
            for tensor in images_iter:
                if not tensor.is_cuda:
                    tensor = tensor.cuda()

                # float32 [0,1] → uint8 on GPU
                frame_u8 = (tensor * 255 + 0.5).clamp(0, 255).to(torch.uint8)

                # [H,W,C] → [C,H,W] contiguous
                frame_chw = frame_u8.permute(2, 0, 1).contiguous()

                # Tensor → VPF Surface (RGB planar)
                surf = nvc.Surface.Make(  # type: ignore[name-defined]
                    nvc.PixelFormat.RGB_PLANAR, width, height, gpu_id  # type: ignore[name-defined]
                )
                surf_plane = surf.PlanePtr()
                pnvc.TensorToDptr(  # type: ignore[name-defined]
                    frame_chw,
                    surf_plane.GpuMem(),
                    surf_plane.Width(),
                    surf_plane.Height(),
                    surf_plane.Pitch(),
                    surf_plane.ElemSize(),
                )

                # RGB → NV12 on GPU
                nv12 = to_nv12.Execute(surf, cc_ctx)

                success = encoder.EncodeSingleSurface(nv12, enc_frame)
                if success:
                    f.write(bytearray(enc_frame))

                total += 1
                pbar.update(1)

                first_frame = False

            while True:
                success = encoder.FlushSinglePacket(enc_frame)
                if success:
                    f.write(bytearray(enc_frame))
                else:
                    break

    except Exception as exc:
        try:
            os.remove(raw_path)
        except OSError:
            pass
        raise RuntimeError(
            f"GPU-direct encoding failed at frame {total}: {exc}"
        ) from exc

    # Mux raw bitstream → container via ffmpeg (copy, no re-encode)
    fmt_flag = {"h264": "h264", "hevc": "hevc", "av1": "ivf"}.get(codec, "h264")
    mux_args = [
        ffmpeg_path, "-v", "error", "-y",
        "-r", str(frame_rate),
        "-f", fmt_flag, "-i", raw_path,
        "-c:v", "copy",
    ]

    if video_format.get("save_metadata", "False") != "False":
        meta_path = _prepare_metadata_file(video_metadata)
        if meta_path:
            mux_args += ["-i", meta_path, "-map", "0:v", "-map_metadata", "1",
                         "-movflags", "use_metadata_tags"]

    mux_args.append(file_path)
    try:
        subprocess.run(mux_args, env=env, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("Muxing failed: %s", exc.stderr.decode(*ENCODE_ARGS))
        # Fall back: rename raw file so user at least has the data
        os.rename(raw_path, file_path)
        return total

    try:
        os.remove(raw_path)
    except OSError:
        pass
    return total


# ═══════════════════════════════════════════════════════════════════════════
# The Node
# ═══════════════════════════════════════════════════════════════════════════
audio_extensions = ["mp3", "mp4", "wav", "ogg"]


class FastVideoCombine:
    """Optimised drop-in replacement for VHS Video Combine.

    Tier selection is automatic:
      • VPF / PyNvVideoCodec detected + NVENC format → GPU-direct  (fastest)
      • otherwise → batched GPU→CPU with threaded ffmpeg pipe      (fast)
      • pre_pass / gifski formats → legacy per-frame generator     (compatible)
    """

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [["lossless", "BOOLEAN", {"default": True}]]
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (floatOrInt, {"default": 8, "min": 1, "step": 1}),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (
                    ["image/gif", "image/webp"] + ffmpeg_formats,
                    {"formats": format_widgets},
                ),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
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
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢"
    FUNCTION = "combine_video"

    # ------------------------------------------------------------------ #

    def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix: str = "AnimateDiff",
        format: str = "image/gif",
        pingpong: bool = False,
        save_output: bool = True,
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

        # --- optional VAE decode ----------------------------------------
        if vae is not None:
            downscale = getattr(vae, "downscale_ratio", 8)
            w = images.size(-1) * downscale
            h = images.size(-2) * downscale
            fpb = (1920 * 1080 * 16) // (w * h) or 1

            def _batched_vae(imgs, _vae, n):
                for batch in _batched_iter(iter(imgs), n):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from _vae.decode(image_batch)

            images = _batched_vae(images, vae, fpb)
            first_image = next(images)
            images = itertools.chain([first_image], images)
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)

        # --- output paths -----------------------------------------------
        output_dir = (
            folder_paths.get_output_directory() if save_output
            else folder_paths.get_temp_directory()
        )
        full_output_folder, filename, _, subfolder, _ = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )
        output_files: list[str] = []

        # --- metadata ---------------------------------------------------
        metadata = PngInfo()
        video_metadata: dict = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get("workflow", {}).get("extra", {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        # --- meta_batch continuation ------------------------------------
        if meta_batch is not None and unique_id in meta_batch.outputs:
            counter, output_process = meta_batch.outputs[unique_id]
        else:
            max_counter = 0
            matcher = re.compile(
                rf"{re.escape(filename)}_(\d+)\D*\..+", re.IGNORECASE
            )
            for existing in os.listdir(full_output_folder):
                m = matcher.fullmatch(existing)
                if m:
                    max_counter = max(max_counter, int(m.group(1)))
            counter = max_counter + 1
            output_process = None

        # --- save first frame as PNG (for metadata) ---------------------
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get("VHS_MetadataImage", True) is not False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path, pnginfo=metadata, compress_level=4,
            )
        output_files.append(file_path)

        # ===============================================================
        #  IMAGE FORMATS  (gif / webp — Pillow path, same as VHS)
        # ===============================================================
        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise RuntimeError("Pillow formats are not compatible with batched output")
            image_kwargs: dict = {}
            if format_ext == "gif":
                image_kwargs["disposal"] = 2
            if format_ext == "webp":
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {
                    36867: datetime.datetime.now().isoformat(" ")[:19]
                }
                image_kwargs["exif"] = exif
                image_kwargs["lossless"] = kwargs.get("lossless", True)
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)

            def _frames_gen(imgs):
                for i in imgs:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))

            frames = _frames_gen(images)
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs,
            )
            output_files.append(file_path)

        # ===============================================================
        #  VIDEO FORMATS  (ffmpeg path — OPTIMISED)
        # ===============================================================
        else:
            if ffmpeg_path is None:
                raise ProcessLookupError(
                    "ffmpeg is required for video outputs and could not be found.\n"
                    "Install imageio-ffmpeg, place ffmpeg in the working directory, "
                    "or add it to PATH."
                )

            if manual_format_widgets is not None:
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)

            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
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
                    image = image.permute((2, 0, 1))
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1, 2, 0))

                images = map(_pad, images)
                dimensions = (
                    -first_image.shape[1] % dim_alignment + first_image.shape[1],
                    -first_image.shape[0] % dim_alignment + first_image.shape[0],
                )
                logger.warning("Padding applied for dimension alignment")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])

            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames - 2
                    pbar.total = num_frames

            if loop_count > 0:
                loop_args = [
                    "-vf", f"loop=loop={loop_count}:size={num_frames}"
                ]
            else:
                loop_args = []

            # --- pixel format -------------------------------------------
            if video_format.get("input_color_depth", "8bit") == "16bit":
                is_16bit = True
                if has_alpha:
                    i_pix_fmt = "rgba64"
                else:
                    i_pix_fmt = "rgb48"
            else:
                is_16bit = False
                if has_alpha:
                    i_pix_fmt = "rgba"
                else:
                    i_pix_fmt = "rgb24"

            channels = 4 if has_alpha else 3

            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)

            bitrate_arg: list[str] = []
            bitrate = video_format.get("bitrate")
            if bitrate is not None:
                suffix = "M" if video_format.get("megabit") == "True" else "K"
                bitrate_arg = ["-b:v", f"{bitrate}{suffix}"]

            args = [
                ffmpeg_path, "-v", "error",
                "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                "-color_range", "pc",
                "-colorspace", "rgb",
                "-color_primaries", "bt709",
                "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                "-s", f"{dimensions[0]}x{dimensions[1]}",
                "-r", str(frame_rate),
                "-i", "-",
            ] + loop_args

            env = os.environ.copy()
            if "environment" in video_format:
                env.update(video_format["environment"])

            # Decide which encoding path to use:
            #   - pre_pass / gifski / meta_batch continuation → legacy generator
            #   - everything else → optimised batched encoder
            use_legacy = (
                "pre_pass" in video_format
                or "gifski_pass" in video_format
                or output_process is not None  # meta_batch continuation
                or meta_batch is not None       # meta_batch first run
                or is_16bit                     # 16-bit needs tensor_to_shorts path
            )

            total_frames_output = 0

            if use_legacy:
                # ── Legacy per-frame path (VHS-compatible) ──────────────
                if is_16bit:
                    images_conv = map(tensor_to_shorts, images)
                else:
                    images_conv = map(tensor_to_bytes, images)
                images_bytes = map(lambda x: x.tobytes(), images_conv)

                if "pre_pass" in video_format:
                    if meta_batch is not None:
                        raise RuntimeError(
                            "Formats with pre_pass are incompatible with Batch Manager."
                        )
                    images_bytes = [b"".join(images_bytes)]
                    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                    in_args_len = args.index("-i") + 2
                    pre_pass_args = args[:in_args_len] + video_format["pre_pass"]
                    merge_filter_args(pre_pass_args)
                    try:
                        subprocess.run(
                            pre_pass_args, input=images_bytes[0], env=env,
                            capture_output=True, check=True,
                        )
                    except subprocess.CalledProcessError as exc:
                        raise RuntimeError(
                            "ffmpeg pre-pass failed:\n" + exc.stderr.decode(*ENCODE_ARGS)
                        )

                if "inputs_main_pass" in video_format:
                    in_args_len = args.index("-i") + 2
                    args = (
                        args[:in_args_len]
                        + video_format["inputs_main_pass"]
                        + args[in_args_len:]
                    )

                if output_process is None:
                    if "gifski_pass" in video_format:
                        format = "image/gif"
                        output_process = gifski_process(
                            args, dimensions, frame_rate, video_format,
                            file_path, env,
                        )
                        audio = None
                    else:
                        args += video_format["main_pass"] + bitrate_arg
                        merge_filter_args(args)
                        output_process = ffmpeg_process_legacy(
                            args, video_format, video_metadata, file_path, env,
                        )
                    output_process.send(None)
                    if meta_batch is not None:
                        meta_batch.outputs[unique_id] = (counter, output_process)

                for image in images_bytes:
                    pbar.update(1)
                    output_process.send(image)

                if meta_batch is not None:
                    try:
                        from ComfyUI_VideoHelperSuite.videohelpersuite.utils import requeue_workflow
                    except ImportError:
                        try:
                            from videohelpersuite.utils import requeue_workflow
                        except ImportError:
                            requeue_workflow = None
                    if requeue_workflow is not None:
                        requeue_workflow(
                            (meta_batch.unique_id, not meta_batch.has_closed_inputs)
                        )

                if meta_batch is not None and not meta_batch.has_closed_inputs:
                    return {
                        "ui": {"unfinished_batch": [True]},
                        "result": ((save_output, []),),
                    }

                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id, None)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()

            else:
                # ── Optimised path (Tier 3 → Tier 1 fallback) ──────────
                if "inputs_main_pass" in video_format:
                    in_args_len = args.index("-i") + 2
                    args = (
                        args[:in_args_len]
                        + video_format["inputs_main_pass"]
                        + args[in_args_len:]
                    )

                gpu_result = None
                if _gpu_encoder_name and not is_16bit:
                    try:
                        gpu_result = encode_gpu_direct(
                            images_iter=images,
                            file_path=file_path,
                            width=dimensions[0],
                            height=dimensions[1],
                            frame_rate=frame_rate,
                            video_format=video_format,
                            video_metadata=video_metadata,
                            pbar=pbar,
                            env=env,
                            bitrate_arg=bitrate_arg,
                        )
                    except RuntimeError:
                        raise
                    except Exception as exc:
                        logger.warning("GPU encoder unavailable: %s", exc)
                        gpu_result = None

                if gpu_result is not None:
                    total_frames_output = gpu_result
                    logger.info(
                        "GPU-direct encode (%s): %d frames via %s",
                        _gpu_encoder_name,
                        total_frames_output,
                        _nvenc_codec_name(video_format),
                    )
                else:
                    ffmpeg_args = list(args)
                    ffmpeg_args += video_format["main_pass"] + bitrate_arg
                    merge_filter_args(ffmpeg_args)
                    ffmpeg_args = _inject_cuda_colorspace(ffmpeg_args)

                    batch_sz = _optimal_batch_size(
                        dimensions[1], dimensions[0], channels,
                    )

                    total_frames_output = encode_ffmpeg_batched(
                        args=ffmpeg_args,
                        video_format=video_format,
                        video_metadata=video_metadata,
                        file_path=file_path,
                        env=env,
                        images_iter=images,
                        pbar=pbar,
                        batch_size=batch_sz,
                    )

            output_files.append(file_path)

            # --- audio muxing -------------------------------------------
            a_waveform = None
            if audio is not None:
                try:
                    a_waveform = audio["waveform"]
                except Exception:
                    pass
            if a_waveform is not None:
                output_file_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_audio_path = os.path.join(full_output_folder, output_file_audio)
                if "audio_pass" not in video_format:
                    logger.warning("Format lacks audio_pass — using libopus")
                    video_format["audio_pass"] = ["-c:a", "libopus"]

                ch = audio["waveform"].size(1)
                min_dur = total_frames_output / frame_rate + 1
                if video_format.get("trim_to_audio", "False") != "False":
                    apad = []
                else:
                    apad = ["-af", f"apad=whole_dur={min_dur}"]

                mux_args = [
                    ffmpeg_path, "-v", "error", "-n",
                    "-i", file_path,
                    "-ar", str(audio["sample_rate"]),
                    "-ac", str(ch),
                    "-f", "f32le", "-i", "-",
                    "-c:v", "copy",
                ] + video_format["audio_pass"] + apad + ["-shortest", output_audio_path]

                audio_data = (
                    audio["waveform"].squeeze(0).transpose(0, 1).numpy().tobytes()
                )
                merge_filter_args(mux_args, "-af")
                try:
                    res = subprocess.run(
                        mux_args, input=audio_data, env=env,
                        capture_output=True, check=True,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        "Audio mux failed:\n" + exc.stderr.decode(*ENCODE_ARGS)
                    )
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_audio_path)
                file = output_file_audio

        # --- keep-intermediate cleanup ----------------------------------
        if extra_options.get("VHS_KeepIntermediate", True) is False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)

        preview = {
            "filename": file,
            "subfolder": subfolder,
            "type": "output" if save_output else "temp",
            "format": format,
            "frame_rate": frame_rate,
            "workflow": first_image_file,
            "fullpath": output_files[-1],
        }
        if num_frames == 1 and "png" in format and "%03d" in file:
            preview["format"] = "image/png"
            preview["filename"] = file.replace("%03d", "001")
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════
NODE_CLASS_MAPPINGS = {
    "FastVideoCombine": FastVideoCombine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVideoCombine": "Video Combine (Fast) ⚡",
}
