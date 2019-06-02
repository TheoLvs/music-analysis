#! /usr/bin/python
# -*- coding:utf-8 -*-


# Usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
import datetime 

from plotly.offline import iplot,init_notebook_mode
import plotly.graph_objs as go

import IPython.display as ipd


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
- https://musicinformationretrieval.com/index.html
"""

KEYS = ["C","C#","D","D#","E","F","F#","G","Ab","A","Bb","B"]



class Music:
    def __init__(self,path,load_fast = True,duration = None,**kwargs):
        
        self.path = path
        self.load(path,load_fast = load_fast,duration = duration,**kwargs)

        # Compute all features
        self.compute_spectrogram()
        self.compute_tempo()
        self.compute_duration()
        self.compute_chromagram()
        self.compute_key()

    def __repr__(self):
        """String representation
        """
        return f"Music(path='{self.path}')"


    def load(self,path,load_fast = True,duration = None,**kwargs):
        """Load music object using librosa library
        Reference at https://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load
        """

        res_type = "kaiser_fast" if load_fast else "kaiser_best"
        self.waveform,self.sampling_rate = librosa.load(path,res_type = res_type,duration = duration,**kwargs)
        print(f"... Loaded '{path}' file")


    def compute_spectrogram(self):
        # Compute the spectrogram and convert to dB
        if not hasattr(self,"spectrogram"):
            self.spectrogram = melspectrogram(self.waveform,self.sampling_rate)
            self.spectrogram_db = power_to_db(self.spectrogram,ref = np.max)  
            print("... Computed spectrogram")  


    def compute_chromagram(self,energy = False):
        """Compute the chromagram
        """    

        if not hasattr(self,"chromagram"):
            if energy:
                S = np.abs(librosa.stft(self.waveform))
                self.chromagram = librosa.feature.chroma_stft(S=S, sr=self.sampling_rate)
            else:
                self.chromagram = librosa.feature.chroma_stft(y=self.waveform, sr=self.sampling_rate)

            print("... Computed chromagram")  


    def compute_tempo(self):
        """Compute tempo and beat frames with the default beat tracker
        """

        # Run the default beat tracker
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.waveform, sr=self.sampling_rate)
        print(f"... Estimated tempo: {self.tempo:.2f} beats per minute")

        # Convert the beat frames to time
        self.beat_times = librosa.frames_to_time(self.beat_frames, sr=self.sampling_rate)


    def compute_duration(self):
        """Compute duration of the music in seconds
        """
    
        # Get music duration
        self.duration = librosa.core.get_duration(y=self.waveform, sr=self.sampling_rate)
        print(f"... Music duration is {self.get_duration(as_str = True)}")


    def compute_key(self):
        """Compute key detection
        TODO add https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd
        """

        # Compute chromagram if not already done
        self.compute_chromagram()

        # Compute keys summary
        self.keys_summary = pd.DataFrame({"key":KEYS,"intensity":self.chromagram.sum(axis = 1)})
        self.keys_summary["intensity"] = self.keys_summary["intensity"] / self.keys_summary["intensity"].max()

        # Compute main key with greedy algorithm
        self.main_key = KEYS[self._get_main_key_index(self.keys_summary)]
        self.is_major = self._is_major(self.keys_summary)
        self.key = self.main_key + ("" if self.is_major else "m")

        print(f"... Detected key: {self.key}")


    def _is_major(self,summary):
        return (summary.loc[summary["key"].isin(self._get_thirds(summary))]
                ["intensity"]
                .reset_index(drop = True)
                .idxmax()
               ) == 1

    def _get_thirds(self,summary):
        key_index = self._get_main_key_index(summary)
        return (KEYS*2)[key_index+3:key_index+5]

    @staticmethod
    def _get_main_key_index(summary):
        return summary["intensity"].idxmax()


    def play(self,with_clicks = False):
        if not with_clicks:
            return ipd.Audio(self.path,rate = self.sampling_rate)
        else:
            clicks = librosa.clicks(self.beat_times, sr=self.sampling_rate, length=len(self.waveform))
            audio = self.waveform + clicks
            return ipd.Audio(audio,rate = self.sampling_rate)


    def show_spectrogram(self,**kwargs):
        """Display the spectrogram
        TODO add plotly or bokeh to dynamically explore the sound
        """
        # Show spectrogram
        plt.figure(figsize = (15,4))
        specshow(self.spectrogram_db,x_axis = "time",**kwargs)
        plt.colorbar()
        plt.tight_layout()
        plt.show()


    def show_waveform(self):
        plt.figure(figsize=(15, 4))
        librosa.display.waveplot(self.waveform, sr=self.sampling_rate)
        plt.tight_layout()
        plt.show()


    def show_chromagram(self,**kwargs):
        """Display the chromagram
        TODO add plotly or bokeh to dynamically explore the sound
        """
        
        # Show chromagram        
        plt.figure(figsize=(15, 4))
        librosa.display.specshow(self.chromagram, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()


    def show_keys(self):

        data = [go.Scatterpolar(
          r = self.keys_summary["intensity"],
          theta = self.keys_summary["key"],
          fill = 'toself'
        )]

        layout = go.Layout(
          title = f"Key: {self.key}",
          polar = dict(
            radialaxis = dict(
              visible = True,
              range = [0, self.keys_summary["intensity"].max()]
            )
          ),
          showlegend = False
        )

        fig = {"data":data,"layout":layout}
        iplot(fig)



    def get_tonality(self):
        pass


    def get_duration(self,as_str = True):
        """Get duration of the music in seconds
        Will compute the duration if not computed before
        """
        if not hasattr(self,"duration"):
            self.compute_duration()

        if as_str:
            return str(datetime.timedelta(seconds=int(self.duration)))
        else:
            return self.duration



    def describe(self):
        """Get all features such as 
        - Tempo
        - Tonality
        - Length
        - Atmosphere
        """

        pass






