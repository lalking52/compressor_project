import torch

# Default STFT params tuned for better quality
DEFAULT_N_FFT = 2048
DEFAULT_HOP = 128


def wav_to_stft(wav, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP, return_phase=False):
    # Use Hann window to reduce spectral leakage
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    )
    mag = torch.log1p(stft.abs())
    if return_phase:
        phase = torch.angle(stft)
        return mag, phase
    return mag


def istft_from_log_mag_phase(log_mag, phase, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP):
    mag = torch.expm1(log_mag)
    window = torch.hann_window(n_fft, device=log_mag.device)
    complex_spec = mag * torch.exp(1j * phase)
    wav = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, window=window)
    return wav


def griffin_lim(log_mag, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP, iters=80):
    mag = torch.expm1(log_mag)
    angle = torch.rand_like(mag) * 2 * torch.pi
    window = torch.hann_window(n_fft, device=log_mag.device)
    for _ in range(iters):
        complex_spec = mag * torch.exp(1j * angle)
        wav = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, window=window)
        est = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        angle = torch.angle(est)
    complex_spec = mag * torch.exp(1j * angle)
    wav = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, window=window)
    return wav


def mag_phase_from_complex(spec):
    """Split complex STFT into log-magnitude and phase."""
    mag = torch.log1p(spec.abs())
    phase = torch.angle(spec)
    return mag, phase


def complex_from_mag_phase(log_mag, phase):
    """Combine log-magnitude and phase into complex STFT tensor."""
    mag = torch.expm1(log_mag)
    return mag * torch.exp(1j * phase)
