import os
import librosa
import soundfile as sf

# Define input and output directories
input_folder = r'C:\safechick\stereo'  # Path to your stereo WAV files
output_folder = r'C:\safechick\mono'  # Folder to save the converted mono files
os.makedirs(output_folder, exist_ok=True)

# Function to convert audio to mono
def convert_to_mono(input_path, output_path):
    try:
        # Load the stereo audio file (mono=False to load in stereo if available)
        print(f"Loading {input_path}...")
        y, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Convert to mono by averaging the stereo channels
        y_mono = librosa.to_mono(y)
        
        # Save the mono file as a new WAV file
        print(f"Saving converted file to {output_path}...")
        sf.write(output_path, y_mono, sr)
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Loop over all wav files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):  # Check if it's a wav file
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')
        
        # Convert and save as mono
        convert_to_mono(input_path, output_path)
        print(f'Converted {filename} to mono and saved as {output_path}')
    else:
        print(f"Skipping non-wav file: {filename}")
