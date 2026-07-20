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
    