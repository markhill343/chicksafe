import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Directories
audio_folder = 'C:\safechick\mono'  # Input folder with audio files
spectrogram_folder = 'C:\safechick\spectogram'  # Output folder for spectrograms
os.makedirs(spectrogram_folder, exist_ok=True)

# Process each audio file in the folder
for filename in os.listdir(audio_folder):
    if filename.endswith('.wav'):
        print(f"Processing file: {filename}")
        audio_path = os.path.join(audio_folder, filename)
        
        try:
            # Load the audio file
            y, sr = librosa.load(audio_path, sr=None)
            print(f"Loaded {filename}, sample rate: {sr}, duration: {len(y)/sr} seconds")
            
            # Generate Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Plot the spectrogram
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel-frequency Spectrogram of {filename}')
            plt.tight_layout()
            
            # Save the spectrogram as a PNG file
            spectrogram_path = os.path.join(spectrogram_folder, f'{os.path.splitext(filename)[0]}.png')
            plt.savefig(spectrogram_path)
            plt.close()  # Close the plot to free memory
            
            print(f"Spectrogram saved to {spectrogram_path}")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
