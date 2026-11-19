import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def extract_features(audio_path, sr=22050, n_mfcc=13, n_mels=128):
    """
    Extract MFCC and Mel-Spectrogram features from an audio file.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    sr : int
        Sample rate (default: 22050 Hz)
    n_mfcc : int
        Number of MFCCs to extract (default: 13)
    n_mels : int
        Number of Mel bands (default: 128)
    
    Returns:
    --------
    dict : Dictionary containing MFCCs and Mel-Spectrogram
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Extract Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return {
            'filename': os.path.basename(audio_path),
            'mfcc': mfccs,
            'mel_spectrogram': mel_spec_db,
            'sample_rate': sr,
            'duration': len(y) / sr
        }
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

def process_batch(folder_path, sr=22050, n_mfcc=13, n_mels=128, save_visualizations=True):
    """
    Process a batch of audio files and extract features.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing audio files
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCCs
    n_mels : int
        Number of Mel bands
    save_visualizations : bool
        Whether to save visualization plots
    
    Returns:
    --------
    list : List of dictionaries containing features for each file
    """
    folder = Path(folder_path)
    audio_files = list(folder.glob('*.mp3')) + list(folder.glob('*.wav'))
    
    if not audio_files:
        print(f"No audio files found in {folder_path}")
        return []
    
    print(f"Found {len(audio_files)} audio files")
    
    features_list = []
    
    # Create output directory for visualizations
    if save_visualizations:
        vis_dir = Path('feature_visualizations')
        vis_dir.mkdir(exist_ok=True)
    
    for idx, audio_file in enumerate(audio_files, 1):
        print(f"Processing {idx}/{len(audio_files)}: {audio_file.name}")
        
        features = extract_features(str(audio_file), sr=sr, n_mfcc=n_mfcc, n_mels=n_mels)
        
        if features:
            features_list.append(features)
            
            # Save visualizations
            if save_visualizations:
                visualize_features(features, vis_dir)
    
    return features_list

def visualize_features(features, output_dir):
    """
    Create and save visualizations of MFCC and Mel-Spectrogram.
    
    Parameters:
    -----------
    features : dict
        Dictionary containing extracted features
    output_dir : Path
        Directory to save visualizations
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot MFCCs
    img1 = librosa.display.specshow(features['mfcc'], 
                                     x_axis='time', 
                                     sr=features['sample_rate'],
                                     ax=axes[0])
    axes[0].set_title(f"MFCC - {features['filename']}")
    axes[0].set_ylabel('MFCC Coefficients')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f')
    
    # Plot Mel-Spectrogram
    img2 = librosa.display.specshow(features['mel_spectrogram'],
                                     x_axis='time',
                                     y_axis='mel',
                                     sr=features['sample_rate'],
                                     ax=axes[1])
    axes[1].set_title(f"Mel-Spectrogram - {features['filename']}")
    axes[1].set_ylabel('Mel Frequency')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / f"{Path(features['filename']).stem}_features.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_path}")

def save_features_to_numpy(features_list, output_dir='feature_arrays'):
    """
    Save extracted features as numpy arrays.
    
    Parameters:
    -----------
    features_list : list
        List of feature dictionaries
    output_dir : str
        Directory to save numpy arrays
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for features in features_list:
        filename_base = Path(features['filename']).stem
        
        # Save MFCCs
        mfcc_path = output_path / f"{filename_base}_mfcc.npy"
        np.save(mfcc_path, features['mfcc'])
        
        # Save Mel-Spectrogram
        mel_path = output_path / f"{filename_base}_mel_spec.npy"
        np.save(mel_path, features['mel_spectrogram'])
    
    print(f"\nSaved feature arrays to {output_dir}/")

def get_statistics(features_list):
    """
    Calculate and print statistics of extracted features.
    
    Parameters:
    -----------
    features_list : list
        List of feature dictionaries
    """
    if not features_list:
        print("No features to analyze")
        return
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION STATISTICS")
    print("="*60)
    
    total_duration = sum(f['duration'] for f in features_list)
    
    print(f"\nTotal files processed: {len(features_list)}")
    print(f"Total audio duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    for features in features_list:
        print(f"\n{features['filename']}:")
        print(f"  Duration: {features['duration']:.2f}s")
        print(f"  MFCC shape: {features['mfcc'].shape}")
        print(f"  Mel-Spectrogram shape: {features['mel_spectrogram'].shape}")
        print(f"  MFCC mean: {np.mean(features['mfcc']):.2f}")
        print(f"  Mel-Spec mean: {np.mean(features['mel_spectrogram']):.2f} dB")

if __name__ == "__main__":
    # Configuration
    AUDIO_FOLDER = 'samples'
    SAMPLE_RATE = 22050
    N_MFCC = 13  # Number of MFCC coefficients
    N_MELS = 128  # Number of Mel bands
    
    print("Starting batch audio feature extraction...")
    print(f"Audio folder: {AUDIO_FOLDER}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"MFCC coefficients: {N_MFCC}")
    print(f"Mel bands: {N_MELS}")
    print()
    
    # Process all audio files
    features = process_batch(
        folder_path=AUDIO_FOLDER,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        save_visualizations=True
    )
    
    # Save features as numpy arrays
    if features:
        save_features_to_numpy(features)
        
        # Print statistics
        get_statistics(features)
        
        print("\n" + "="*60)
        print("Feature extraction completed successfully!")
        print("="*60)
    else:
        print("No features extracted. Please check your audio folder.")