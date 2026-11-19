import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load audio file (same folder)
audio_path = 'eg1.wav'
y, sr = librosa.load(audio_path, sr=16000)

# Extract MFCC features
n_mfcc = 40
mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=n_mfcc,
    n_fft=2048,
    hop_length=512
)

print("MFCC Shape:", mfcc.shape)

# Visualize MFCC
plt.figure(figsize=(12, 5))
librosa.display.specshow(
    mfcc,
    sr=sr,
    x_axis='time'
)

plt.colorbar(label='MFCC Coefficients')
plt.title('MFCC Features')
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()
