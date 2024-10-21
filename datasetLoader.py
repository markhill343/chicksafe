import os
import torch
import librosa
from torch.utils.data import Dataset

class AudioTextDataset(Dataset):
    def __init__(self, audio_folder, labels):
        self.audio_folder = audio_folder
        self.labels = labels
        self.audio_files = os.listdir(audio_folder)
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load the audio file
        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_folder, audio_file)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Get the corresponding text label
        label = self.labels[audio_file]
        
        return y, label

# Dictionary mapping each audio file to its corresponding label
labels = {
    'Alarm-01.wav': 'Alarm',
    'Alarm-02.wav': 'Alarm',
    # Continue for all other files
    'Squawk-01.wav': 'Squawk',
    'Attack-01.wav': 'Attack',
    # etc.
}
