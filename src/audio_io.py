import shutil
import subprocess
from pathlib import Path
import warnings

import numba_cache  # noqa: F401 ensures numba can cache
import audioread
import librosa
import soundfile as sf

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".audio_cache"
# Cache folder for on-the-fly decoded WAVs to avoid repeated conversions
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _convert_with_ffmpeg(src: Path, dst: Path, sr, mono) -> bool:
    """
    Convert audio using ffmpeg if available. Returns True on success.
    """
    if shutil.which("ffmpeg") is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if mono:
        cmd += ["-ac", "1"]
    if sr is not None:
        cmd += ["-ar", str(sr)]
    cmd += ["-vn", "-acodec", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def _convert_with_afconvert(src: Path, dst: Path):
    """
    Convert an audio file to WAV using macOS afconvert so we can load
    containers (e.g. ALAC/AAC) without requiring ffmpeg.
    """
    if shutil.which("afconvert") is None:
        raise RuntimeError("Cannot decode audio file and afconvert is not available.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["afconvert", str(src), str(dst), "-f", "WAVE", "-d", "LEI16"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def load_audio(path, sr=None, mono=True):
    """
    Load audio with librosa, converting to WAV on the fly if the file is an
    m4a/ALAC container that Librosa cannot decode in this environment.
    """
    try:
        return librosa.load(path, sr=sr, mono=mono)
    except (sf.LibsndfileError, audioread.NoBackendError) as exc:
        src_path = Path(path)
        converted = _CACHE_DIR / f"{src_path.stem}_converted.wav"
        if not converted.exists():
            warnings.warn(f"Converting {src_path.name} to WAV for decoding.")
            try:
                if not _convert_with_ffmpeg(src_path, converted, sr, mono):
                    _convert_with_afconvert(src_path, converted)
            except Exception as convert_exc:
                raise RuntimeError(
                    "Failed to decode audio file. Install ffmpeg or provide WAV files."
                ) from convert_exc
        # Load the cached WAV version to keep downstream code uniform
        return librosa.load(converted, sr=sr, mono=mono)
