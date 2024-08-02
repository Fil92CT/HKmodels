#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:54:22 2024

@author: filipporiscicalizzio
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

"""SHANNON ENTROPY HEATMAPS_CB"""

def shannon_entropy(dictionary, n):
    """dictionary of opinions; n number of nodes"""
    summation = []

    for cluster_id, count in dictionary.items():
        k = count
        f = k/n
        summation.append(f)
    pk = np.array(summation)

    return pk


def calculate_entropy(opinions, max_distance=0.02):
    data_array = opinions.to_numpy().flatten()
    clusters = []
    assigned = [False] * len(data_array)

    for i in range(len(data_array)):
        if not assigned[i]:
            cluster = [data_array[i]]
            assigned[i] = True
            for j in range(i + 1, len(data_array)):
                if not assigned[j] and np.abs(data_array[j] - data_array[i]) <= max_distance:
                    cluster.append(data_array[j])
                    assigned[j] = True
            clusters.append(cluster)

    cluster_counts = {i: len(cluster) for i, cluster in enumerate(clusters)}
    sn = shannon_entropy(cluster_counts, len(data_array))
    return entropy(sn)

def aggregate_entropies(base_dir):
    entropy_data = {}  # Dictionary to store entropy data
    population_sizes = []  # List to store all population sizes

    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith('CycleR_'):
            population_size = int(complete_dir.split('_')[1])
            population_sizes.append(population_size)  # Add the population size to the list
            path = os.path.join(base_dir, complete_dir)
            
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    opinions = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    ent = calculate_entropy(opinions)
                    # Use a tuple (population_size, confidence_bound) as a key
                    entropy_data.setdefault((population_size, confidence_bound), []).append(ent)

    # Calculate average entropy for each (population_size, confidence_bound) pair
    avg_entropy = {(pop_size, cb): np.mean(values) for (pop_size, cb), values in entropy_data.items()}

    # Return the average entropy data, sorted confidence bounds, and population sizes
    return avg_entropy, sorted(set(k[1] for k in avg_entropy.keys())), sorted(population_sizes)

def create_heatmap(base_dir):
    avg_entropy, confidence_bounds, population_sizes = aggregate_entropies(base_dir)
    heatmap_data = pd.DataFrame(index=confidence_bounds, columns=population_sizes)

    # Populate the heatmap data
    for (pop_size, cb), ent in avg_entropy.items():
        heatmap_data.at[cb, pop_size] = ent

    # Convert all data to numeric
    heatmap_data = heatmap_data.apply(pd.to_numeric)

    sns.heatmap(heatmap_data, annot=True, cmap="viridis")
    plt.xlabel('Population Size')
    plt.ylabel('Confidence Bound')
    plt.title('Mean Shannon Entropy Heatmap')
    plt.show()
    
# Replace with the path to your 'Complete' directory
#list_graphs = [whe]
base_dir = '/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_Base/CycleR/CycleR_0.3'
#graph = base_dir[-1]
create_heatmap(base_dir)