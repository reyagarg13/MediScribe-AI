import noisereduce as nr
import librosa
import soundfile as sf

def reduce_noise(input_file, output_file):
    # Load audio
    y, sr = librosa.load(input_file, sr=None)

    # Reduce noise
    reduced = nr.reduce_noise(y=y, sr=sr)

    # Save cleaned audio
    sf.write(output_file, reduced, sr)

# Example usage
reduce_noise("raw_audio.wav", "clean_audio.wav")
