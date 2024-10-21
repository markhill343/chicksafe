from transformers import CLAPProcessor, CLAPModel
from torch.utils.data import DataLoader, Dataset
import torch
import os
import librosa

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
    'crow-01.wav': 'Crow',
    'crow-02.wav': 'Crow',
    'crow-03.wav': 'Crow',
    'crow-04.wav': 'Crow',
    'crow-05.wav': 'Crow',
    'crow-06.wav': 'Crow',
    'crow-07.wav': 'Crow',
    'crow-08.wav': 'Crow',
    'crow-09.wav': 'Crow',
    'crow-10.wav': 'Crow',
    'crow-11.wav': 'Crow',
    'crow-12.wav': 'Crow',
    'crow-13.wav': 'Crow',
    'crow-14.wav': 'Crow',
    'crow-15.wav': 'Crow',
    'crow-16.wav': 'Crow',
    'crow-17.wav': 'Crow',
    'crow-18.wav': 'Crow',
    'crow-19.wav': 'Crow',
    'crow-20.wav': 'Crow',
    'crow-21.wav': 'Crow',
    'crow-22.wav': 'Crow',
    'crow-23.wav': 'Crow',
    'crow-24.wav': 'Crow',
    'crow-25.wav': 'Crow',
    'crow-26.wav': 'Crow',
    'crow-27.wav': 'Crow',
    'crow-28.wav': 'Crow',
    'crow-29.wav': 'Crow',
    'crow-30.wav': 'Crow',
    'crow-31.wav': 'Crow',
    'crow-32.wav': 'Crow',

    'danger-01.wav': 'Danger',
    'danger-02.wav': 'Danger',
    'danger-03.wav': 'Danger',
    'danger-04.wav': 'Danger',
    'danger-05.wav': 'Danger',
    'danger-06.wav': 'Danger',
    'danger-07.wav': 'Danger',
    'danger-08.wav': 'Danger',
    'danger-09.wav': 'Danger',
    'danger-10.wav': 'Danger',
    'danger-11.wav': 'Danger',
    'danger-12.wav': 'Danger',
    'danger-13.wav': 'Danger',
    'danger-14.wav': 'Danger',

    'raid-01.wav': 'Raid'
}



# Load CLAP model and processor
processor = CLAPProcessor.from_pretrained("huggingface/clap")
model = CLAPModel.from_pretrained("huggingface/clap")

# Prepare your dataset
audio_folder = r'C:\safechick\mono'
dataset = AudioTextDataset(audio_folder=audio_folder, labels=labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set up optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(10):  # Adjust the number of epochs as needed
    model.train()
    for batch in dataloader:
        audio, text_labels = batch
        
        # Preprocess audio and text
        audio_inputs = processor(audios=audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
        text_inputs = processor(text=text_labels, return_tensors="pt", padding=True).to(device)
        
        # Forward pass: get embeddings
        audio_embeds = model.get_audio_features(**audio_inputs)
        text_embeds = model.get_text_features(**text_inputs)
        
        # Compute contrastive loss (or another loss function suitable for your task)
        # Example: Use cosine similarity loss
        loss_fn = torch.nn.CosineEmbeddingLoss()
        target = torch.ones(audio_embeds.size(0)).to(device)  # Labels for matching pairs
        loss = loss_fn(audio_embeds, text_embeds, target)
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1} completed with loss: {loss.item()}")
