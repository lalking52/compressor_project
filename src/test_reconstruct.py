import os
import glob
import sys

import soundfile as sf
import torch
from audio_io import load_audio
from inference import load_model, compress, decompress

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

try:
    model = load_model("results/model_last.pth", device=device)
except FileNotFoundError as exc:
    sys.exit(str(exc))

os.makedirs("results", exist_ok=True)
SAMPLE_RATE = 48000

files = glob.glob("data/**/*.wav", recursive=True) + glob.glob("data/**/*.flac", recursive=True)

for path in files:
    wav, sr = load_audio(path, sr=SAMPLE_RATE, mono=True)
    wav = torch.from_numpy(wav).float()
    # One-pass compress/decompress to verify pipeline works
    blob = compress(model, wav, device=device)
    recon_wav = decompress(model, blob, device=device)
    recon_wav = recon_wav.cpu().numpy()
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join("results", f"reconstructed_{name}.wav")
    sf.write(out_path, recon_wav, SAMPLE_RATE)
