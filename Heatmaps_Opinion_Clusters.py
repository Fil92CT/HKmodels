#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:07:04 2024

@author: filipporiscicalizzio
"""
import os
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

""" HEATMAPS OPINION CLUSTERS"""

def count_opinion_clusters(data_array, max_distance=0.02):
    clusters = []
    for data_point in data_array:
        for cluster in clusters:
            if all(np.abs(data_point - np.array(cluster)) <= max_distance):
                cluster.append(data_point)
                break
        else:
            clusters.append([data_point])
    return len(clusters)

def aggregate_cluster_data(base_dir):
    cluster_data = {}  # Dictionary to store cluster data
    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith("ER_"):
            population_size = int(complete_dir.split('_')[1])
            path = os.path.join(base_dir, complete_dir)
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    data_array = df.to_numpy().flatten()
                    num_clusters = count_opinion_clusters(data_array)
                    cluster_data.setdefault((population_size, confidence_bound), []).append(num_clusters)
    #print(cluster_data)
    return cluster_data

def create_heatmap(cluster_data):
    # Transform the cluster_data into a DataFrame suitable for heatmap
    #print(cluster_data)
    data = []
    for (population_size, confidence_bound), counts in cluster_data.items():
        avg_count = np.mean(counts)
        data.append([population_size, confidence_bound, avg_count])
    
    heatmap_df = pd.DataFrame(data, columns=['Population_Size', 'Confidence_Bound', 'Average_Cluster_Count'])
    heatmap_pivot = heatmap_df.pivot("Confidence_Bound", "Population_Size", "Average_Cluster_Count")

    sns.heatmap(heatmap_pivot, annot=True, cmap="viridis")
    plt.xlabel('Population Size')
    plt.ylabel('Confidence Bound')
    plt.title('Mean Opinion Clusters Heatmap')
    plt.show()

# Replace with the path to your 'Complete' directory
base_dir = '/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/Erdos_Renyi/ER_0.8'
cluster_data = aggregate_cluster_data(base_dir)
create_heatmap(cluster_data)