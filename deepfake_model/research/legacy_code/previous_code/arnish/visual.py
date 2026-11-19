import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# Optional: check current directory and files
print("Current Working Directory:", os.getcwd())
print("Files in this folder:", os.listdir())

# Load audio file (same folder)
audio_path = 'eg1.wav'
y, sr = librosa.load(audio_path)

# Plot waveform
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Waveform')

plt.tight_layout()
plt.show()
