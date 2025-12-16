import random
import torch
from torch.utils.data import Dataset
from audio_io import load_audio
from stft_utils import wav_to_stft

class AudioDataset(Dataset):
    def __init__(self, file_list, sr=None, max_seconds=None):
        """
        file_list: list of audio paths
        sr: target sample rate to enforce (keeps Mel/STFT scales consistent)
        max_seconds: randomly crop audio to this duration to speed up training
        """
        self.files = file_list
        self.sr = sr
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = load_audio(self.files[idx], sr=self.sr, mono=True)  # enforce target SR and mono
        if self.max_seconds is not None:
            max_samples = int(self.max_seconds * sr)
            if max_samples > 0 and len(wav) > max_samples:
                start = random.randint(0, len(wav) - max_samples)
                wav = wav[start : start + max_samples]
        wav = torch.from_numpy(wav).float()
        mag, phase = wav_to_stft(wav, return_phase=True)
        # Stack log-magnitude and phase as two channels for complex prediction
        stft_pair = torch.stack([mag, phase], dim=0)
        return stft_pair
