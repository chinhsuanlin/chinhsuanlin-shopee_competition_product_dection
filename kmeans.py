# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:57:21 2020

@author: Public_1080
"""

import numpy as np
from  sklearn.cluster import KMeans
data = np.load('33.npy', allow_pickle=True)


kmeans = KMeans(n_clusters=2, random_state=0).fit(data[1])
