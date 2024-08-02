#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:40:48 2024

@author: filipporiscicalizzio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

"""PERCENTAGE CORRECT OPINION VS TRUTH-ATTRACTION"""

distance_threshold = 0.02
file_paths = '/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/Erdos_Renyi/ER_0.8'

# Dictionary to store the mean percentage of agents within the specified distance for each confidence bound
mean_percentage_within_distance_by_ta = {}

for folder in os.listdir(file_paths):
    path = os.path.join(file_paths, folder)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.startswith('simulation_'):
                # Extract the confidence bound from the filename
                ta = float(file.split('_')[3])
                t = float(file.split('_')[2])
                df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                final_opinions = df.to_numpy().flatten()
                distance_from_t = np.abs(final_opinions - t)
                within_distance = distance_from_t <= distance_threshold
                percentage_within_distance = np.mean(within_distance) * 100
                    
                if ta not in mean_percentage_within_distance_by_ta:
                    mean_percentage_within_distance_by_ta[ta] = []
                mean_percentage_within_distance_by_ta[ta].append(percentage_within_distance)

# Calculate mean percentage for each confidence bound
for ta, percentages in mean_percentage_within_distance_by_ta.items():
    mean_percentage_within_distance_by_ta[ta] = np.mean(percentages)

# Sort by confidence bound (assuming they are numeric)
truth_attraction = sorted(mean_percentage_within_distance_by_ta.keys())
mean_percentages = [mean_percentage_within_distance_by_ta[ta] for ta in truth_attraction]

plt.figure(figsize=(10, 6))
plt.plot(truth_attraction, mean_percentages, marker='o', linestyle='-')
plt.xlabel('Truth Attraction')
plt.ylabel('Mean Percentage of Agents')
plt.title('Truth Approximation')
plt.grid(True)
plt.show()