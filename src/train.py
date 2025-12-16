import glob
import os
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchaudio
from torch.cuda.amp import autocast, GradScaler
from dataset import AudioDataset
from model_fc_ae import FullyConvAE
from stft_utils import DEFAULT_N_FFT, DEFAULT_HOP, istft_from_log_mag_phase

SAMPLE_RATE = 48000  # keep STFT/mel configs consistent

files = sorted(
    glob.glob("data/**/*.wav", recursive=True) + glob.glob("data/**/*.flac", recursive=True)
)  # recursive lookup to allow nested speaker folders
if not files:
    raise RuntimeError("No training audio found in data/ (wav or flac).")
max_train = int(os.environ.get("MAX_TRAIN_FILES", "20"))
if max_train > 0 and len(files) > max_train:
    random.seed(42)
    random.shuffle(files)
    files = files[:max_train]
    print(f"Subsampled training set to {len(files)} files (MAX_TRAIN_FILES={max_train})")
torch.manual_seed(42)
max_audio_seconds = float(os.environ.get("MAX_AUDIO_SECONDS", "8"))
if max_audio_seconds <= 0:
    max_audio_seconds = None
# Optional random crop to keep batches short and training fast
dataset = AudioDataset(files, sr=SAMPLE_RATE, max_seconds=max_audio_seconds)  # optional random crop for speed

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
wave_loss_weight = 0.3
mel_loss_weight = float(os.environ.get("MEL_LOSS_WEIGHT", "0.003"))
epochs = int(os.environ.get("EPOCHS", "80"))
batch_size = 1  # variable-length STFTs, keep batch=1 to avoid collation errors
print(
    f"Device: {device} | training files: {len(dataset)} | epochs: {epochs} | batch_size: {batch_size}"
)
model_channels = int(os.environ.get("MODEL_CHANNELS", "8"))
model = FullyConvAE(channels=model_channels).to(device)
ckpt_path = "results/model_last.pth"
if os.path.exists(ckpt_path):
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Resumed from {ckpt_path}")
    except Exception as exc:
        print(f"Could not load checkpoint ({exc}), starting fresh.")
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
loss_fn = nn.L1Loss()
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP,
    n_mels=80,
    center=True,
    power=1.0,
).to(device)
mel_transform_mid = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
    center=True,
    power=1.0,
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80).to(device)
mel_loss_fn = nn.L1Loss()
os.makedirs("results", exist_ok=True)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=device.type == "cuda",
)
use_amp = device.type == "cuda"
scaler = GradScaler(enabled=use_amp)
best_loss = float("inf")
best_ckpt_path = "results/model_best.pth"
for epoch in range(epochs):
    total_loss = 0.0
    steps = 0
    for stft_pair in loader:
        stft_pair = stft_pair.to(device)
        with autocast(enabled=use_amp):
            z, pred = model(stft_pair)
            min_f = min(pred.size(-2), stft_pair.size(-2))
            min_t = min(pred.size(-1), stft_pair.size(-1))
            pred = pred[..., :min_f, :min_t]  # align shapes after strided convs
            stft_crop = stft_pair[..., :min_f, :min_t]
            pred_mag = pred[:,0]
            pred_phase = pred[:,1]
            target_mag = stft_crop[:,0]
            target_phase = stft_crop[:,1]
            loss_mag = loss_fn(pred_mag, target_mag)
            loss_phase = loss_fn(pred_phase, target_phase)
            loss_spec = loss_mag + loss_phase
            # Reconstruct waveforms for waveform/mel losses
            wav_pred = istft_from_log_mag_phase(pred_mag[0], pred_phase[0])
            wav_orig = istft_from_log_mag_phase(target_mag[0], target_phase[0])
            loss_wave = torch.mean(torch.abs(wav_pred - wav_orig))

            # Mel loss for perceptual alignment
            mel_pred = amp_to_db(mel_transform(wav_pred.unsqueeze(0)))
            mel_orig = amp_to_db(mel_transform(wav_orig.unsqueeze(0)))
            mel_pred_mid = amp_to_db(mel_transform_mid(wav_pred.unsqueeze(0)))
            mel_orig_mid = amp_to_db(mel_transform_mid(wav_orig.unsqueeze(0)))
            loss_mel = mel_loss_fn(mel_pred, mel_orig) + mel_loss_fn(mel_pred_mid, mel_orig_mid)

            loss = loss_spec + wave_loss_weight * loss_wave + mel_loss_weight * loss_mel
        opt.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item()
        steps += 1
        if steps % 50 == 0:
            print(f"epoch {epoch} step {steps}/{len(loader)} | loss={loss.item():.4f}")
    epoch_loss = total_loss / max(1, steps)
    print(
        f"epoch {epoch}: total={epoch_loss:.4f} spec={loss_spec.item():.4f} wave={loss_wave.item():.4f} mel={loss_mel.item():.4f}"
    )
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"Saved improved checkpoint to {best_ckpt_path}")

torch.save(model.state_dict(), "results/model_last.pth")
