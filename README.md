Audio Compression with Fully Convolutional Autoencoders

Overview
This project trains and evaluates a neural audio compressor. Audio is converted to a complex STFT (log-magnitude + phase as two channels), compressed by a fully convolutional autoencoder, and reconstructed. The goal is to study label-free compression and balance quality (SNR / SI-SDR / LSD) against latent/blob size.

Project Structure
.
├── src/
│   ├── audio_io.py          # Audio loading with on-the-fly conversion if needed
│   ├── stft_utils.py        # STFT/iSTFT helpers for magnitude/phase
│   ├── dataset.py           # PyTorch Dataset returning stacked [log-mag, phase]
│   ├── model_fc_ae.py       # Fully-conv autoencoder with down/up-sampling and bottleneck resblock
│   ├── train.py             # Training loop (complex spec, mel + waveform losses)
│   ├── inference.py         # compress/decompress with latent quantization
│   ├── eval.py              # Metrics: SNR, SI-SDR, LSD, compression/size reporting
│   ├── test_reconstruct.py  # Reconstruct from a checkpoint
│   ├── size_check.py        # Size/compression checks
│   └── numba_cache.py       # Optional numba cache directory setup
├── data/                    # Training wav/flac
├── data_test/               # Test wav/flac
├── results/                 # Checkpoints and recon_*.wav
└── README.md

Model
- Input: two-channel complex STFT (log-magnitude, phase).
- Encoder: Conv2d stack with stride=2 plus a bottleneck ResBlock.
- Decoder: mirrored ConvTranspose2d predicting both magnitude and phase to avoid stored phase dependency.
- Quantization: per-channel max-norm, int8/16 (default 8 bits), no bit-packing.

Training
Run:
```
python src/train.py
```
Environment knobs:
- `MAX_TRAIN_FILES` (default 20) — limit number of training files.
- `MAX_AUDIO_SECONDS` (default 8) — random crop for speed (<=0 disables).
- `MODEL_CHANNELS` (default 8) — width; smaller means stronger compression.
- `MEL_LOSS_WEIGHT` (default 0.003) — mel loss weight (lower = softer, less robotic).
- `EPOCHS` (default 80).

Losses: L1 on magnitude/phase, L1 on waveform, mel loss at two resolutions.

Inference and Evaluation
- Reconstruction: `python src/test_reconstruct.py` (requires `results/model_last.pth`).
- Evaluation: `python src/eval.py` saves `recon_*.wav` and prints compression (input/latent, blob size), SNR, SI-SDR, LSD.
- Core API in `src/inference.py`: `compress`/`decompress` with `store_phase` (bool) and `quant_bits` (default 8).

Dataset prep
Use `src/prepare_split.py` to create train/test symlinks without duplicating audio:
```
python src/prepare_split.py --source LibriSpeech/test-clean --train-ratio 0.5 --seed 42
```

Quick smoke test
Verify the pipeline end-to-end:
```
python src/smoke_test.py
```
Takes `data_test/*.wav` (or generates a 1s 440 Hz tone), runs two settings (near-lossless vs degraded), and writes recon/residual to `results/`.

Future directions
- Time-domain encoder/decoder (Conv/TasNet-style) for better perceptual quality at similar bitrate.
- Fully learned phase/complex decoding to reduce reliance on stored phase.
- Perceptual metrics: PESQ/STOI (speech), multi-band LSD/SDR, MOS proxies.
- Variational/VQ latent (codebook or KL) for more controllable compression.
- Multi-resolution or learned filterbanks instead of fixed STFT.

Requirements
- Python 3.8+
- PyTorch
- NumPy
- SciPy
- librosa (optional)
- numba (optional)

```
pip install torch numpy scipy librosa numba
```

License
Intended for research and educational use.
