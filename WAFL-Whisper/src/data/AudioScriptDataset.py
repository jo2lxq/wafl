import os

import torch
import torchaudio
import whisper
from torch.utils.data import Dataset
from torchaudio.transforms import Resample


class AudioScriptDataset(Dataset):
    def __init__(self, data_dir, tokenizer, transform=None, sample_rate=16000):
        self.audio_paths = [
            os.path.join(f"{data_dir}/audio", fname) for fname in sorted(os.listdir(f"{data_dir}/audio"))
        ]
        self.scripts = [
            os.path.join(f"{data_dir}/script", fname) for fname in sorted(os.listdir(f"{data_dir}/script"))
        ]
        self.transform = transform
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        script_path = self.scripts[idx]

        # Load script file
        with open(script_path, "r", encoding="utf-8") as file:
            text = file.read().strip()

        # Load audio file
        audio = self._load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,  # Audio data
            "dec_input_ids": text,  # Input to decoder
            "labels": labels,  # Ground truth data
            "audio_path": audio_path,
            "script_path": script_path,
        }

    def _load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            resampler = Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        return waveform
