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

"""
Inspiration : 
- https://github.com/vivjay30/pychorus
- https://github.com/librosa/librosa
"""



#=============================================================================================================
# MAIN MUSIC LOADER
#=============================================================================================================



class Music(object):
    def __init__(self,path):
        
        self.path = path
        self.load(path)

    def __repr__(self):
        """String representation
        """
        return f"Music(path='{path}')"


    def load(self,path):
        """Load music object using librosa library
        """
        self.y,self.sr = librosa.load(path)
        print(f"... Loaded '{path}' file")


    def spectrogram(self,**kwargs):
        """Compute spectrogram of the music and display it
        """

        # Compute the spectrogram and convert to dB
        if not hasattr(self,"s"):
            self.s = melspectrogram(self.y,self.sr)
            self.sdb = power_to_db(self.s,ref = np.max)

        # Show spectrogram
        plt.figure(figsize = (15,4))
        specshow(self.sdb,x_axis = "time",**kwargs)
        plt.colorbar()
        plt.tight_layout()
        plt.show()







