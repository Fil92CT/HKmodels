#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 21:11:35 2024

@author: filipporiscicalizzio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

"""PERCENTAGE CORRECT OPINION VS CONFIDENCE BOUND"""

distance_threshold = 0.02
file_paths = '/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/Erdos_Renyi/ER_0.8'

# Dictionary to store the mean percentage of agents within the specified distance for each confidence bound
mean_percentage_within_distance_by_cb = {}

for folder in os.listdir(file_paths):
    path = os.path.join(file_paths, folder)
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.startswith('simulation_'):
                # Extract the confidence bound from the filename
                cb = float(file.split('_')[-1].replace('.csv', ''))
                t = float(file.split('_')[2])
                df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                final_opinions = df.to_numpy().flatten()
                distance_from_t = np.abs(final_opinions - t)
                within_distance = distance_from_t <= distance_threshold
                percentage_within_distance = np.mean(within_distance) * 100
                    
                if cb not in mean_percentage_within_distance_by_cb:
                    mean_percentage_within_distance_by_cb[cb] = []
                mean_percentage_within_distance_by_cb[cb].append(percentage_within_distance)

# Calculate mean percentage for each confidence bound
for cb, percentages in mean_percentage_within_distance_by_cb.items():
    mean_percentage_within_distance_by_cb[cb] = np.mean(percentages)

# Sort by confidence bound (assuming they are numeric)
confidence_bounds = sorted(mean_percentage_within_distance_by_cb.keys())
mean_percentages = [mean_percentage_within_distance_by_cb[cb] for cb in confidence_bounds]

plt.figure(figsize=(10, 6))
plt.plot(confidence_bounds, mean_percentages, marker='o', linestyle='-')
plt.xlabel('Confidence Bound')
plt.ylabel('Mean Percentage of Agents')
plt.title('Truth Approximation')
plt.grid(True)
plt.show()