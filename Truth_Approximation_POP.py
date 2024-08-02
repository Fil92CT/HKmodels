#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:56:50 2024

@author: filipporiscicalizzio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

"""TRUTH APPROXIMATION AGAINST POPULATION SIZE"""

distance_threshold = 0.02

file_paths = '/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/Erdos_Renyi/ER_0.2' 
mean_percentage_within_distance = {}

for folder in os.listdir(file_paths):
    path = os.path.join(file_paths, folder)
    if os.path.isdir(path):
        population_size = folder.split('_')[1]
        #print(path)
        # Dictionary to store the mean percentage of agents within the specified distance for each population size
        percentages_within_distance = []
        for file in os.listdir(path):
            
            if file.startswith('simulation_'):
                #print('YES')
                t = float(file.split('_')[2])
                df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                final_opinions = df.to_numpy().flatten()
                for opinion in final_opinions:
                    # Simulate the calculation of percentage of agents within the specified distance from t
                    distance_from_t = np.abs(final_opinions - t)
                    within_distance = distance_from_t <= distance_threshold
                    percentage_within_distance = np.mean(within_distance) * 100
                    percentages_within_distance.append(percentage_within_distance)
                # Calculate the mean percentage for this population size
                mean_percentage_within_distance[population_size] = np.mean(percentages_within_distance)
            #print(mean_percentage_within_distance)

# Convert population sizes to integers for sorting and plotting
population_sizes = [int(size) for size in mean_percentage_within_distance.keys()]
population_sizes.sort()  # Sort population sizes to ensure correct plotting order
mean_percentages = [mean_percentage_within_distance[str(size)] for size in population_sizes]

plt.figure(figsize=(10, 6))
plt.plot(population_sizes, mean_percentages, marker='o', linestyle='-')
plt.xlabel('Population Size')
plt.ylabel('Mean Percentage of Agents Within Distance')
plt.title('Mean Percentage of Agents Within 0.02 Distance from t vs. Population Size')
plt.grid(True)
plt.show()
            # Load the CSV file
            
            