import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from CNN_NETs import CNN_Nets
import torch


G_path = '/Users/sepehrbe/My_Drive/DataSources/SkinCare'
G_saved = G_path+ '/Saved/'
G_Grid = G_path+ '/Saved/Grid_Search_Results'

files = list(os.listdir(G_Grid))
f = 'CNN_280x210_Model3_21_Jun22_15-01'
GridSrach = pickle.load(open(G_Grid + '/' + f , 'rb'))
