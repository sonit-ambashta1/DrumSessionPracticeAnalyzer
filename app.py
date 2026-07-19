import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa.beat import beat_track
import librosa.display

if __name__=="__main__":
    y, sr = librosa.load(
        "mixed-grooves-straight_91bpm.wav",
        sr=None
    )
    
    librosa.display.waveshow(y)
    plt.show()
    
    tempo, beats = beat_track(y=y, sr=sr)
    print(f"TEMPO: {tempo}")
    print(f"BEAT FRAMES: {beats}")
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"BEAT TIMES: {beat_times}")
    
    sound_frames = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=None)
    print(f"SOUND FRAMES: {sound_frames}")
    sound_times = librosa.frames_to_time(sound_frames, sr=sr)
    print(f"SOUND TIMES: {sound_times}")
    
    # transform for time conversion
    
    
    # # Waveform
    # plt.figure(figsize=(12, 4))
    # librosa.display.waveshow(y, sr=sr)
    # plt.title("Waveform")
    # plt.savefig("waveform.png")
    # plt.close()

    # # Spectrogram
    # D = librosa.stft(y)
    # S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # plt.figure(figsize=(12, 6))
    # librosa.display.specshow(
    #     S_db,
    #     sr=sr,
    #     x_axis="time",
    #     y_axis="log"
    # )
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Spectrogram")
    # plt.savefig("spectra.png")
    # plt.close()
