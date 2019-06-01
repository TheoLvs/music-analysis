#! /usr/bin/python
# -*- coding:utf-8 -*-


# Usual libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook

# Custom libraries
import librosa
from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa import power_to_db

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
Inspiration : 
- https://github.com/vivjay30/pychorus
- https://github.com/librosa/librosa
"""



class Music:
    def __init__(self,path,load_fast = True,**kwargs):
        
        self.path = path
        self.load(path,load_fast = load_fast,**kwargs)
        self.compute_spectrogram()
        self.compute_tempo()

    def __repr__(self):
        """String representation
        """
        return f"Music(path='{self.path}')"


    def load(self,path,load_fast = True,**kwargs):
        """Load music object using librosa library
        Reference at https://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
        """

        res_type = "kaiser_fast" if load_fast else "kaiser_best"
        self.waveform,self.sampling_rate = librosa.load(path,res_type = res_type,**kwargs)
        print(f"... Loaded '{path}' file")


    def compute_spectrogram(self):
        # Compute the spectrogram and convert to dB
        if not hasattr(self,"spectrogram"):
            self.spectrogram = melspectrogram(self.waveform,self.sampling_rate)
            self.spectrogram_db = power_to_db(self.spectrogram,ref = np.max)  
            print("... Computed spectrogram")      


    def compute_tempo(self):
        """Compute tempo and beat frames with the default beat tracker
        """

        # Run the default beat tracker
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.waveform, sr=self.sampling_rate)
        print(f"... Estimated tempo: {self.tempo:.2f} beats per minute")

        # Convert the beat frames to time
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sampling_rate)


    def show_spectrogram(self,**kwargs):
        """Compute spectrogram of the music and display it
        TODO add plotly or bokeh to dynamically explore the sound
        """

        self.compute_spectrogram()

        # Show spectrogram
        plt.figure(figsize = (15,4))
        specshow(self.spectrogram_db,x_axis = "time",**kwargs)
        plt.colorbar()
        plt.tight_layout()
        plt.show()



    def get_tonality(self):
        pass


    def describe(self):
        """Get all features such as 
        - Tempo
        - Tonality
        - Length
        - Atmosphere
        """
    
        # Get music duration
        self.duration = librosa.core.get_duration(y=self.waveform, sr=self.sampling_rate)







