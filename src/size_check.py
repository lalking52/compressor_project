import glob
import os
import sys

import torch

from audio_io import load_audio
from inference import load_model, compress

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = load_model("results/model_last.pth", device=device)
except FileNotFoundError as exc:
    sys.exit(str(exc))
SAMPLE_RATE = 48000

files = glob.glob("data_test/**/*.wav", recursive=True) + glob.glob("data_test/**/*.flac", recursive=True)

for path in files:
    original_size = os.path.getsize(path)
    wav, sr = load_audio(path, sr=SAMPLE_RATE, mono=True)
    wav_t = torch.from_numpy(wav).float()
    # Disable phase storage to see lower bound on size
    blob = compress(model, wav_t, device=device, quant_bits=8, store_phase=False)
    compressed_size = len(blob)
    # For fair comparison with PCM WAV (16-bit), estimate uncompressed size
    pcm_size = int(len(wav) * 2)  # 2 bytes per sample, mono
    ratio_file = original_size / compressed_size if compressed_size > 0 else float("inf")
    ratio_pcm = pcm_size / compressed_size if compressed_size > 0 else float("inf")
    print(
        f"{os.path.basename(path)} | orig_file={original_size} bytes | pcm16_est={pcm_size} bytes | compressed={compressed_size} bytes | ratio_vs_file={ratio_file:.2f}x | ratio_vs_pcm={ratio_pcm:.2f}x"
    )
