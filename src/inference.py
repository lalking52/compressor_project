import os
import pickle
import zlib

import numpy as np
import torch

from model_fc_ae import FullyConvAE
from stft_utils import (
    griffin_lim,
    istft_from_log_mag_phase,
    wav_to_stft,
    complex_from_mag_phase,
    mag_phase_from_complex,
)


def _dtype_for_bits(quant_bits: int):
    """
    Pick the minimal integer dtype that can store the requested signed range.
    No bit-packing is done; <=8 bits use int8, 9–16 bits use int16.
    """
    if not (2 <= quant_bits <= 16):
        raise ValueError("quant_bits must be in [2, 16]")
    return torch.int8 if quant_bits <= 8 else torch.int16


def load_model(model_path, device="cpu"):
    """
    Load model and infer channel count from checkpoint to avoid mismatches.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint {model_path} not found. Train a model first (python src/train.py)."
        )
    state = torch.load(model_path, map_location=device)
    first_weight = next(v for k, v in state.items() if k.endswith("encoder.0.weight"))
    channels = first_weight.shape[0]
    model = FullyConvAE(channels=channels)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def compress(model, wav, device="cpu", quant_bits=8, store_phase=True):
    """
    Encode waveform into compressed blob with per-channel max-norm scale and int quantization.
    No bit-packing is performed: <=8 bits store as int8, 9–16 bits store as int16.
    store_phase=False reduces size (uses predicted phase on decode).
    """
    wav = wav.to(device)
    mag, phase = wav_to_stft(wav, return_phase=True)
    stft_pair = torch.stack([mag, phase], dim=0)  # [2, F, T]
    rms = float(wav.abs().mean().item())
    x = stft_pair.unsqueeze(0).to(device)  # [1, 2, F, T]
    with torch.no_grad():
        latent, recon_pair = model(x)
    latent = latent.cpu()
    # Per-channel max scaling keeps values within quantization range
    scale = latent.abs().amax(dim=(2, 3), keepdim=True)
    scale[scale == 0] = 1.0
    q_level = (2 ** (quant_bits - 1)) - 1  # e.g., 127 for int8
    q_dtype = _dtype_for_bits(quant_bits)
    latent_q = torch.clamp((latent / scale) * q_level, -q_level, q_level).round().to(q_dtype)

    # If we store phase separately, quantize the original STFT phase; otherwise rely on predicted phase.
    phase_q = (
        torch.clamp((phase / np.pi) * 32767.0, -32767.0, 32767.0).round().to(torch.int16)
        if store_phase
        else None
    )

    payload = {
        "latent_q": latent_q.numpy(),
        "scale": scale.cpu().numpy().astype(np.float16),
        "shape": stft_pair.shape[1:],  # F, T
        "phase_q": phase_q.cpu().numpy() if phase_q is not None else None,
        "rms": rms,
        "quant_bits": quant_bits,
        "orig_len": int(wav.shape[-1]),
    }
    blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL), level=9)
    return blob


def decompress(model, blob, device="cpu"):
    """
    Decode compressed blob back to waveform using stored phase when available.
    """
    payload = pickle.loads(zlib.decompress(blob))
    latent_q = torch.from_numpy(payload["latent_q"])
    scale = torch.from_numpy(payload["scale"])
    quant_bits = int(payload.get("quant_bits", 8))
    target_shape = payload["shape"]  # F, T
    phase_q_np = payload.get("phase_q")
    target_rms = payload.get("rms")
    target_len = payload.get("orig_len")

    q_level = (2 ** (quant_bits - 1)) - 1
    latent = latent_q.to(torch.float32) * scale / q_level
    latent = latent.to(device)
    with torch.no_grad():
        recon_pair = model.decoder(latent)
    # recon_pair shape [1, 2, F, T]
    recon_pair = recon_pair.squeeze(0)
    if target_shape is not None:
        recon_pair = recon_pair[..., : target_shape[0], : target_shape[1]]
    recon_mag_pred, recon_phase_pred = recon_pair[0], recon_pair[1]
    if phase_q_np is not None:
        phase = torch.from_numpy(phase_q_np).to(recon_mag_pred.device).to(torch.float32)
        phase = phase * (np.pi / 32767.0)
        wav = istft_from_log_mag_phase(recon_mag_pred, phase)
    else:
        # Use predicted phase from the decoder
        wav = istft_from_log_mag_phase(recon_mag_pred, recon_phase_pred)
    if target_rms is not None:
        current_rms = wav.abs().mean()
        if current_rms > 1e-8:
            wav = wav * (target_rms / current_rms)
    if target_len is not None:
        wav = wav[..., :target_len]
    return wav
