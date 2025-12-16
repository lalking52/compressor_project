import glob
import os
import pickle
import random
import zlib
import numpy as np
import soundfile as sf
import sys
import torch
from audio_io import load_audio
from inference import load_model, compress, decompress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = load_model("results/model_last.pth", device=device)
except FileNotFoundError as exc:
    sys.exit(str(exc))
os.makedirs("results", exist_ok=True)
SAMPLE_RATE = 48000

def snr(x, y):
    # Standard SNR in dB comparing reference and reconstruction
    return 10 * np.log10(np.sum(x**2) / np.sum((x-y)**2))

def si_sdr(target, estimate, eps=1e-8):
    """
    Scale-invariant SDR. Higher is better.
    """
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    target_energy = np.sum(target ** 2)
    if target_energy <= eps:
        return float("nan")
    projection = np.sum(target * estimate) * target / (target_energy + eps)
    noise = estimate - projection
    return 10 * np.log10((np.sum(projection ** 2) + eps) / (np.sum(noise ** 2) + eps))


def log_spectral_distance(x, y, n_fft=512, hop=128, eps=1e-12):
    """
    RMS log-spectral distance in dB between two waveforms. Lower is better.
    """
    xt = torch.from_numpy(x).float()
    yt = torch.from_numpy(y).float()
    window = torch.hann_window(n_fft)
    X = torch.stft(xt, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    Y = torch.stft(yt, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag_x = torch.clamp(X.abs(), min=eps)
    mag_y = torch.clamp(Y.abs(), min=eps)
    lsd = torch.sqrt(torch.mean((20 * torch.log10(mag_x) - 20 * torch.log10(mag_y)) ** 2))
    return float(lsd.item())

files = glob.glob("data_test/**/*.wav", recursive=True) + glob.glob("data_test/**/*.flac", recursive=True)
max_test = int(os.environ.get("MAX_TEST_FILES", "20"))
if max_test > 0 and len(files) > max_test:
    random.seed(42)
    random.shuffle(files)
    files = files[:max_test]
    print(f"Subsampled test set to {len(files)} files (MAX_TEST_FILES={max_test})")

for path in files:
    wav, sr = load_audio(path, sr=SAMPLE_RATE, mono=True)
    wav_t = torch.from_numpy(wav).float()
    # Compress and immediately decompress to measure fidelity and compression
    blob = compress(model, wav_t, device=device)
    payload = pickle.loads(zlib.decompress(blob))
    input_shape = payload["shape"]
    latent_shape = payload["latent_q"].shape
    input_vals = int(np.prod(input_shape))
    latent_vals = int(np.prod(latent_shape))
    recon = decompress(model, blob, device=device).cpu().numpy()
    min_len = min(len(wav), len(recon))
    wav = wav[:min_len]
    recon = recon[:min_len]
    # Compute quality metrics
    snr_val = snr(wav, recon)
    si_sdr_val = si_sdr(wav, recon)
    lsd_val = log_spectral_distance(wav, recon)
    comp_ratio = input_vals / max(latent_vals, 1)
    print(
        f"{path} | input STFT {input_shape} ({input_vals} vals) -> latent {latent_shape} ({latent_vals}) "
        f"-> recon len {len(recon)} | blob {len(blob)} bytes | compression≈{comp_ratio:.2f}x | "
        f"SNR {snr_val:.2f} dB | SI-SDR {si_sdr_val:.2f} dB | LSD {lsd_val:.2f} dB"
    )
    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join("results", f"recon_{base}.wav")
    # Save reconstructed audio for listening checks
    sf.write(out_path, recon, SAMPLE_RATE)
