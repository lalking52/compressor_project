import glob
import math
import os
import sys

import soundfile as sf
import torch

from audio_io import load_audio
from inference import compress, decompress, load_model
from model_fc_ae import FullyConvAE
from stft_utils import istft_from_log_mag_phase, wav_to_stft

SAMPLE_RATE = 48000
TONE_FREQ = 440.0
TONE_DURATION = 1.0  # seconds


def make_tone(freq=TONE_FREQ, duration=TONE_DURATION, sr=SAMPLE_RATE):
    t = torch.linspace(0, duration, int(sr * duration), dtype=torch.float32, device="cpu")
    return 0.2 * torch.sin(2 * math.pi * freq * t)  # avoid clipping


def snr(x, y):
    num = torch.sum(x * x)
    denom = torch.sum((x - y) ** 2)
    if denom == 0:
        return float("inf")
    return 10 * torch.log10(num / denom).item()


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ckpt_path = "results/model_last.pth"
if os.path.exists(ckpt_path):
    model = load_model(ckpt_path, device=device)
    print(f"Loaded checkpoint from {ckpt_path}")
else:
    model = FullyConvAE(channels=64).to(device).eval()
    print("No checkpoint found; using randomly initialized model (expected low SNR).")

os.makedirs("results", exist_ok=True)

files = sorted(
    glob.glob("data_test/**/*.wav", recursive=True) + glob.glob("data_test/**/*.flac", recursive=True)
)
if not files:
    print("No data_test/*.wav found; using synthetic 440 Hz tone for smoke test.")
    files = [None]

for path in files:
    if path is None:
        wav = make_tone().to(device)
        label = "tone"
    else:
        wav_np, _ = load_audio(path, sr=SAMPLE_RATE, mono=True)
        wav = torch.from_numpy(wav_np).float().to(device)
        label = os.path.splitext(os.path.basename(path))[0]

    mag, phase = wav_to_stft(wav, return_phase=True)
    baseline = istft_from_log_mag_phase(mag, phase)

    # Two settings: near-lossless (phase stored) and degraded (no phase + 4-bit quant)
    configs = [
        {"name": "phase", "quant_bits": 8, "store_phase": True},
        {"name": "no_phase_q4", "quant_bits": 4, "store_phase": False},
    ]

    for cfg in configs:
        # Round-trip through compressor for each configuration
        blob = compress(
            model,
            wav,
            device=device,
            quant_bits=cfg["quant_bits"],
            store_phase=cfg["store_phase"],
        )
        recon = decompress(model, blob, device=device)

        min_len = min(wav.shape[-1], baseline.shape[-1], recon.shape[-1])
        wav_c = wav[..., :min_len].cpu()
        baseline_c = baseline[..., :min_len].cpu()
        recon_c = recon[..., :min_len].cpu()

        baseline_snr = snr(wav_c, baseline_c)
        model_snr = snr(wav_c, recon_c)

        residual = recon_c - wav_c
        l1 = torch.mean(torch.abs(residual)).item()
        l2 = torch.sqrt(torch.mean(residual ** 2)).item()
        peak = torch.max(torch.abs(residual)).item()

        print(f"[{label}/{cfg['name']}] Baseline STFT/iSTFT SNR (no model): {baseline_snr:.2f} dB")
        print(f"[{label}/{cfg['name']}] Model recon SNR (compress→decompress): {model_snr:.2f} dB")
        print(
            f"[{label}/{cfg['name']}] Residual stats | L1: {l1:.6f} | L2: {l2:.6f} | peak: {peak:.6f}"
        )

        recon_path = f"results/smoke_recon_{label}_{cfg['name']}.wav"
        residual_path = f"results/smoke_residual_{label}_{cfg['name']}.wav"
        sf.write(recon_path, recon_c.numpy(), SAMPLE_RATE)
        sf.write(residual_path, residual.numpy(), SAMPLE_RATE)
        print(f"Wrote {recon_path} and {residual_path}\n")
