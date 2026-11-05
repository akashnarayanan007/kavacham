"""Ingest a video, reduce it for SAM, and run the whole pipeline on it, for the dashboard.

Standard library only at import time, so app.py can use it without the pipeline's
dependencies. OpenCV is loaded only by the fallback reducer, and the pipeline scripts
run as separate processes with the venv's Python.

A run folder, runs/<timestamp>/, holds:
    original.<ext>       the video as received (a hard link when it came from this folder)
    video.mp4            the reduced copy SAM tracks
    original_frame.jpg   first frame of each, to compare sharpness
    reduced_frame.jpg
    reduce_info.json     both videos' details and the ffmpeg command
and, after tracking, every pipeline output.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from common import VISION_MODEL, sam_cost

PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = Path("runs")
LOCK_FILE = RUNS_DIR / ".active.json"

TARGET_SHORT_SIDE = 720  # 720x1280 for portrait phone video, 1280x720 for landscape
TARGET_FPS = 10  # SAM bills per frame, and 10 fps is enough to follow road users
CRF = 18  # near-transparent x264 quality, so SAM sees clean edges on small road users
PRESET = "slow"
MIN_SECONDS = 5
COSTLY_SECONDS = 120
MAX_UPLOAD_BYTES = 1_000_000_000
LOW_BITRATE_MBPS = 1.5

# (label, script, arguments, share of the progress bar done when it finishes)
PIPELINE = [
    ("Tracking every road user with SAM 3.1", "sam_track.py", ["--full"], 0.45),
    ("Finding potential conflicts", "detect.py", [], 0.55),
    (f"Checking top dangerous conflicts with {VISION_MODEL}", "verify.py", ["--top", "20", "--fast", "--use-cache"], 0.85),
    ("Drawing each conflict's tracked video", "generate_88_trackings.py", [], 0.95),
    ("Drawing the hotspot map", "hotspot.py", [], 1.0),
]
STEP_TIMEOUT_SECONDS = 45 * 60

COST_PATTERNS = {
    "sam_track.py": re.compile(r"Estimated cost: \$([\d.]+)"),  # exact: SAM bills per frame
    "verify.py": re.compile(r"Cost: \$([\d.]+)"),
    "hotspot.py": re.compile(r"Cost: \$([\d.]+)"),
}


class PipelineError(Exception):
    """A pipeline step failed. Carries the step's label and the end of its log."""

    def __init__(self, label, log):
        super().__init__(f"{label} failed")
        self.label = label
        self.log = log


class RunInProgress(Exception):
    """Another pipeline run holds the lock, so starting a second would pay twice."""


def probe(path):
    """Size, codec, frame rate, frame count, length and bitrate from ffprobe, or None."""
    if not shutil.which("ffprobe"):
        return None
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames"
                          ":format=duration,bit_rate,size",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        stream, container = data["streams"][0], data["format"]
        fps = float(Fraction(stream["r_frame_rate"]))
        duration = float(container["duration"])
        frames = stream.get("nb_frames", "N/A")
        frames = int(frames) if frames.isdigit() else round(duration * fps)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, ZeroDivisionError):
        return None
    return {
        "codec": stream.get("codec_name"),
        "width": stream["width"],
        "height": stream["height"],
        "fps": fps,
        "frames": frames,
        "seconds": duration,
        "size_mb": int(container.get("size", 0)) / 1_000_000,
        "bitrate_mbps": int(container.get("bit_rate", 0)) / 1_000_000,
    }


def describe(info):
    """'700.2 MB, 2160x3840, 60 fps, 68.9 sec, 4,133 frames' for the UI and logs."""
    return (f'{info["size_mb"]:.1f} MB, {info["width"]}x{info["height"]}, {info["fps"]:g} fps, '
            f'{info["seconds"]:.1f} sec, {info["frames"]:,} frames')


def validate(info, size_bytes):
    """Problems with an original, as (level, message): level 'error' stops the run."""
    problems = []
    if info["seconds"] < MIN_SECONDS:
        problems.append(("error", f'The video is only {info["seconds"]:.1f} sec. Film at least '
                                  f"{MIN_SECONDS} sec of the junction."))
    if info["seconds"] > COSTLY_SECONDS:
        problems.append(("warning", f'The video is {info["seconds"]:.0f} sec, longer than the '
                                    f"{COSTLY_SECONDS} sec planned for, so tracking and "
                                    "verification cost more."))
    if size_bytes > MAX_UPLOAD_BYTES:
        problems.append(("warning", "Files over 1 GB are too big to upload through the browser. "
                                    "Put the file in the Kavacham folder and pick it there."))
    return problems


def ingest(source, run_dir):
    """Keep the original in the run folder. A file already on this disk is hard-linked,
    so a 700 MB video doesn't take 700 MB more."""
    source = Path(source)
    target = Path(run_dir) / f"original{source.suffix.lower()}"
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def reduced_size(width, height, short_side=TARGET_SHORT_SIDE):
    """Shrink the shorter side to short_side, keeping the aspect ratio and even sizes.
    Never enlarges."""
    scale = min(1.0, short_side / min(width, height))
    return round(width * scale / 2) * 2, round(height * scale / 2) * 2


def _reduce_ffmpeg(input_path, output_path, size, fps):
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(input_path),
        "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos,fps={fps}",
        "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])
    return command


def _reduce_opencv(input_path, output_path, size, fps):
    """Slower fallback when ffmpeg is missing. Writes MPEG-4, which SAM accepts."""
    import cv2

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    index, next_time = 0, 0.0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        # Keep the first frame at or after each 1/fps step of source time.
        if index / source_fps >= next_time:
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4))
            next_time += 1 / fps
        index += 1
    capture.release()
    writer.release()
    return ["opencv", "resize", f"{size[0]}x{size[1]}", "lanczos4", f"{fps} fps", "mp4v"]


def _first_frame(video, image):
    """Save frame 0 as a JPEG, for comparing the original's sharpness with the reduced copy."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", "select=eq(n\\,0)", "-frames:v", "1", "-q:v", "2", str(image)],
                   capture_output=True)
    return image if Path(image).exists() else None


def reduce_for_sam(original, run_dir, source_label=None):
    """Reduce the original to video.mp4 at 720p and 10 fps, and record what was done.

    Returns the contents of reduce_info.json.
    """
    run_dir = Path(run_dir)
    original_info = probe(original)
    if original_info is None:
        raise RuntimeError(f"Could not read {original}. Is it a video, and is ffprobe installed?")
    size = reduced_size(original_info["width"], original_info["height"])
    output = run_dir / "video.mp4"

    method = "ffmpeg"
    try:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found")
        command = _reduce_ffmpeg(original, output, size, TARGET_FPS)
    except RuntimeError:
        method = "opencv"
        command = _reduce_opencv(original, output, size, TARGET_FPS)

    reduced = probe(output)
    frames = None
    if shutil.which("ffmpeg"):
        frames = (_first_frame(original, run_dir / "original_frame.jpg"),
                  _first_frame(output, run_dir / "reduced_frame.jpg"))
    info = {
        "source": source_label or str(original),
        "method": method,
        "crf": CRF if method == "ffmpeg" else None,
        "command": command,
        "original": original_info,
        "reduced": reduced,
        "frames_compared": bool(frames and all(frames)),
        "sam_cost_usd": round(sam_cost(reduced["frames"]), 4) if reduced else None,
        "low_bitrate": bool(reduced and reduced["bitrate_mbps"] < LOW_BITRATE_MBPS),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "reduce_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def downscale_video(input_path, output_path):
    """Reduce a video outside any run folder. Kept for scripts that only need the copy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = probe(input_path)
    size = reduced_size(original["width"], original["height"])
    if shutil.which("ffmpeg"):
        _reduce_ffmpeg(input_path, output_path, size, TARGET_FPS)
    else:
        _reduce_opencv(input_path, output_path, size, TARGET_FPS)
    return {"original": original, "reduced": probe(output_path), "path": str(output_path)}


def new_run_dir():
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    return run_dir


def _process_alive(pid):
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # query limited info
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)  # on POSIX, signal 0 only checks the process exists
    except OSError:
        return False
    return True


def active_run():
    """The run folder of a pipeline run still in progress, or None."""
    try:
        lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if _process_alive(lock["pid"]):
        return lock["run_dir"]
    LOCK_FILE.unlink(missing_ok=True)  # left by a process that has exited
    return None


def _acquire_lock(run_dir):
    RUNS_DIR.mkdir(exist_ok=True)
    current = active_run()
    if current:
        raise RunInProgress(current)
    try:
        handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunInProgress(active_run() or "another run") from None
    with os.fdopen(handle, "w", encoding="utf-8") as lock:
        json.dump({"pid": os.getpid(), "run_dir": str(run_dir),
                   "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, lock)


def pipeline_python():
    """The venv's Python, since the pipeline needs packages the dashboard doesn't."""
    for candidate in (PROJECT_DIR / "venv" / "Scripts" / "python.exe",
                      PROJECT_DIR / "venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _run_step(script, args, run_dir, env, on_line):
    """Run one script in the run folder, passing each output line to on_line."""
    process = subprocess.Popen(
        [pipeline_python(), str(PROJECT_DIR / script), *args],
        cwd=run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    timer = threading.Timer(STEP_TIMEOUT_SECONDS, process.kill)
    timer.start()
    lines = []
    try:
        for line in process.stdout:
            if line.strip():
                lines.append(line.rstrip())
                on_line(line.rstrip())
        return process.wait(), lines
    finally:
        timer.cancel()


def run_backend_pipeline(video_path, run_dir, progress_callback, env_overrides=None):
    """Run every pipeline step on video_path inside run_dir.

    Only one run may be in progress at a time: a second raises RunInProgress, so two
    browser sessions can't pay for the same work twice. progress_callback(fraction,
    label, line) is called when a step starts (line=None) and for each line it prints.
    Raises PipelineError if a step fails, leaving what finished in run_dir. Returns the
    output paths and the costs, also saved to costs.json.
    """
    run_dir = Path(run_dir)
    _acquire_lock(run_dir)
    try:
        if Path(video_path).resolve() != (run_dir / "video.mp4").resolve():
            shutil.copyfile(video_path, run_dir / "video.mp4")
        env = {**os.environ, **(env_overrides or {}), "PYTHONIOENCODING": "utf-8",
               "PYTHONUNBUFFERED": "1"}

        costs, done = {}, 0.0
        for label, script, args, finished in PIPELINE:
            progress_callback(done, label, None)
            code, lines = _run_step(
                script, args, run_dir, env,
                lambda line, at=done, label=label: progress_callback(at, label, line))
            if code != 0:
                raise PipelineError(label, lines[-20:])
            pattern = COST_PATTERNS.get(script)
            matches = [pattern.search(line) for line in lines] if pattern else []
            found = [float(match.group(1)) for match in matches if match]
            if found:
                costs[script.removesuffix(".py")] = found[-1]
            done = finished

        summary = {f"{name}_usd": value for name, value in costs.items()}
        summary["total_usd"] = round(sum(costs.values()), 4)
        summary["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (run_dir / "costs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        progress_callback(1.0, "Done", None)
    finally:
        LOCK_FILE.unlink(missing_ok=True)

    names = ("tracks.json", "potential_conflicts.json", "verified_conflicts.json", "hotspot.png")
    return {"run_dir": str(run_dir), "costs": summary,
            **{name: str(run_dir / name) for name in names}}
