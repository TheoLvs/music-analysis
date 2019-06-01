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
import sys

from plotly.offline import iplot,init_notebook_mode
import plotly.graph_objs as go

# Custom libraries
import librosa
from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa import power_to_db

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Custom library
from .music import Music


KEYS = ["C","C#","D","D#","E","F","F#","G","Ab","A","Bb","B"]


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




class Playlist:
    def __init__(self,paths = None,folder = None,load_fast = True,n = None,**kwargs):

        # Prepare paths list
        if folder is not None:
            self.paths = [os.path.join(folder,x) for x in os.listdir(folder)]
        else:
            self.paths = paths
        self.paths = [path for path in self.paths if path.endswith(".mp3")]
        assert len(self.paths) > 0

        # Select a subsection of files
        if n is not None:
            self.paths = self.paths[:n]

        # Read all files
        self.data = []
        for path in tqdm_notebook(self.paths):
            with HiddenPrints():
                music = Music(path = path,load_fast = load_fast)
            self.data.append(music)

    def __repr__(self):
        """String representation
        """
        return f"Playlist(n={len(self)})"


    def __len__(self):
        return len(self.data)


    def __iter__(self):
        return iter(self.data)


    def __getitem__(self,key):
        return self.data[key]


    def compute_features(self):
        pass


    def show_keys(self):

        data = [
            go.Scatterpolar(
              r = music.keys_summary["intensity"],
              theta = music.keys_summary["key"],
              fill = 'toself',
              name = music.path
            ) for music in self
        ]

        layout = go.Layout(
          polar = dict(
            radialaxis = dict(
              visible = True,
              range = [0, 1]
            )
          ),
          showlegend = False
        )

        fig = {"data":data,"layout":layout}
        iplot(fig)