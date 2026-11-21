"""Writing refocused video without per-frame image files.

Frames are quantised on the GPU, copied to the CPU and streamed to one ffmpeg process that encodes the whole clip; the
pipe is fed from a background thread, so encoding overlaps rendering. The encoder is chosen automatically from what
this ffmpeg build has: NVENC (h264_nvenc, on the GPU) if a GPU accepts it, else libx264, libopenh264 or mpeg4.

  from flashrefocus.video import VideoWriter
  with VideoWriter("out.mp4", fps=30) as writer:
      for frame in frames:                      # (1, 3, H, W) or (b, 3, H, W) float in [0, 1], any device
          writer.write(refocuser.render(...))

ffmpeg: $FLASHREFOCUS_FFMPEG, else imageio-ffmpeg's binary if installed, else `ffmpeg` on the PATH.
"""
import os
import queue
import shutil
import subprocess
import threading

import torch

_WORKS = {}


def ffmpeg_exe():
    exe = os.environ.get("FLASHREFOCUS_FFMPEG")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FileNotFoundError("ffmpeg not found: install it, pip install imageio-ffmpeg, or set FLASHREFOCUS_FFMPEG")
    return exe


def encoder_works(codec, exe=None):
    """True if this ffmpeg can actually encode with `codec` here (a short test encode, cached): a build may list an
    encoder whose library or GPU is missing."""
    exe = exe or ffmpeg_exe()
    key = (exe, codec)
    if key not in _WORKS:
        r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=size=256x256:rate=30",
                            "-t", "0.2", "-c:v", codec, "-pix_fmt", "yuv420p", "-f", "null", "-"], capture_output=True)
        _WORKS[key] = r.returncode == 0
    return _WORKS[key]


def available_encoders(exe=None):
    """Video encoder names of this ffmpeg build."""
    exe = exe or ffmpeg_exe()
    out = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return {line.split()[1] for line in out.splitlines() if line.startswith(" V")}


def pick_encoder(exe=None):
    exe = exe or ffmpeg_exe()
    names = available_encoders(exe)
    for name in ("h264_nvenc", "libx264", "libopenh264", "mpeg4"):
        if name in names and encoder_works(name, exe):
            return name
    raise RuntimeError(f"no working H.264 or MPEG-4 encoder in {exe}")


def to_uint8_frames(x):
    """(b, 3, H, W) or (3, H, W) float in [0, 1] -> (b, H, W, 3) uint8 numpy, quantised on the tensor's device."""
    x = x[None] if x.dim() == 3 else x
    return (x.float().clamp(0, 1) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()


class VideoWriter:
    """Streams frames to one ffmpeg encode. codec: "auto" (see pick_encoder) or an ffmpeg encoder name; quality: CRF
    for libx264, CQ for NVENC (lower is better). audio_from: a video file whose audio track is copied in."""

    def __init__(self, path, fps=30.0, codec="auto", quality=18, preset=None, audio_from=None, queue_frames=8):
        self.path, self.fps, self.quality, self.audio_from = str(path), fps, quality, audio_from
        self.exe = ffmpeg_exe()
        self.codec = pick_encoder(self.exe) if codec == "auto" else codec
        self.preset = preset or {"h264_nvenc": "p5", "libx264": "veryfast"}.get(self.codec)
        self.proc, self.size = None, None
        self.q = queue.Queue(maxsize=queue_frames)
        self.error = None
        self.thread = threading.Thread(target=self._pump, daemon=True)

    def _start(self, w, h):
        rate = f"{self.fps}"
        cmd = [self.exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", rate, "-i", "-"]
        if self.audio_from:
            cmd += ["-i", str(self.audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"]
        q = {"h264_nvenc": ["-cq", str(self.quality)], "libx264": ["-crf", str(self.quality)],
             "libopenh264": ["-b:v", "20M"], "mpeg4": ["-q:v", "2"]}.get(self.codec, [])
        preset = ["-preset", self.preset] if self.preset else []
        cmd += ["-c:v", self.codec, *preset, *q, "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.size = (w, h)
        self.thread.start()

    def _pump(self):
        try:
            while True:
                arr = self.q.get()
                if arr is None:
                    break
                self.proc.stdin.write(arr.tobytes())
        except Exception as e:                                   # pragma: no cover
            self.error = e

    def write(self, frames):
        """Append frames: a tensor (b, 3, H, W) / (3, H, W) in [0, 1], or uint8 arrays (b, H, W, 3) / (H, W, 3)."""
        if self.error:
            raise self.error
        arr = to_uint8_frames(frames) if torch.is_tensor(frames) else frames
        arr = arr[None] if arr.ndim == 3 else arr
        if self.proc is None:
            self._start(arr.shape[2], arr.shape[1])
        assert (arr.shape[2], arr.shape[1]) == self.size, "every frame must have the size of the first"
        for a in arr:
            self.q.put(a)

    def close(self):
        if self.proc is None:
            return
        self.q.put(None)
        self.thread.join()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        if self.error:
            raise self.error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
