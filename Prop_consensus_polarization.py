#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:14:00 2024

@author: filipporiscicalizzio
"""

import os
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

"""PLOT PROPORTION CONSENSUS OR POLARIZATION"""

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

def aggregate_consensus_data(base_dir):
    consensus_data = {}
    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith(f'{graph}_'):
            path = os.path.join(base_dir, complete_dir)
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    data_array = df.to_numpy().flatten()
                    num_clusters = count_opinion_clusters(data_array)
                    consensus = 1 if num_clusters == 1 else 0
                    consensus_data.setdefault(confidence_bound, []).append(consensus)

    # Calculate the proportion of simulations reaching consensus for each confidence bound
    consensus_proportion = {cb: sum(consensus) / len(consensus) for cb, consensus in consensus_data.items()}
    return consensus_proportion

def aggregate_perfect_consensus_data(base_dir):
    consensus_data = {}
    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith(f'{graph}_'):
            path = os.path.join(base_dir, complete_dir)
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    data_array = df.to_numpy().flatten()
                    all_identical = np.all(data_array == data_array[0])
                    consensus = 1 if all_identical else 0
                    consensus_data.setdefault(confidence_bound, []).append(consensus)

    # Calculate the proportion of simulations reaching consensus for each confidence bound
    consensus_proportion = {cb: sum(consensus) / len(consensus) for cb, consensus in consensus_data.items()}
    return consensus_proportion

def aggregate_polarization_data(base_dir):
    polarization_data = {}
    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith(f'{graph}_'):
            path = os.path.join(base_dir, complete_dir)
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    data_array = df.to_numpy().flatten()
                    num_clusters = count_opinion_clusters(data_array)
                    polarization = 1 if num_clusters == 2 else 0
                    polarization_data.setdefault(confidence_bound, []).append(polarization)

    # Calculate the proportion of simulations reaching polarization for each confidence bound
    polarization_proportion = {cb: sum(polarization) / len(polarization) for cb, polarization in polarization_data.items()}
    return polarization_proportion

def aggregate_fragmentation_data(base_dir):
    fragmentation_data = {}
    for complete_dir in os.listdir(base_dir):
        if complete_dir.startswith(f'{graph}_'):
            path = os.path.join(base_dir, complete_dir)
            for file in os.listdir(path):
                if file.startswith('simulation_'):
                    confidence_bound = float(file.split('_')[-1].replace('.csv', ''))
                    df = pd.read_csv(os.path.join(path, file)).iloc[-1]
                    data_array = df.to_numpy().flatten()
                    num_clusters = count_opinion_clusters(data_array)
                    fragmentation = 1 if num_clusters >= 3 else 0
                    fragmentation_data.setdefault(confidence_bound, []).append(fragmentation)

    # Calculate the proportion of simulations reaching fragmentation for each confidence bound
    fragmentation_proportion = {cb: sum(fragmentation) / len(fragmentation) for cb, fragmentation in fragmentation_data.items()}
    return fragmentation_proportion


def plot_consensus(consensus_proportion):
    # Sorting the data by confidence bound for plotting
    sorted_data = sorted(consensus_proportion.items())
    confidence_bounds, proportions = zip(*sorted_data)

    plt.plot(confidence_bounds, proportions, marker='o')
    #plt.xlabel('Confidence Bound')
    #plt.ylabel('Proportion of Simulations with Consensus')
    #plt.title('Proportion of Simulations Ending with Consensus by Confidence Bound')
    plt.grid(True)
    plt.show()

def plot_perfect_consensus(consensus_proportion):
    # Sorting the data by confidence bound for plotting
    sorted_data = sorted(consensus_proportion.items())
    confidence_bounds, proportions = zip(*sorted_data)

    plt.plot(confidence_bounds, proportions, marker='o', color='blue')
    #plt.xlabel('Confidence Bound')
    #plt.ylabel('Proportion of Simulations with Consensus')
    #plt.title('Proportion of Simulations Ending with Perfect Consensus by Confidence Bound')
    plt.grid(True)
    plt.show()
    
def plot_polarization(polarization_proportion):
    # Sorting the data by confidence bound for plotting
    sorted_data = sorted(polarization_proportion.items())
    confidence_bounds, proportions = zip(*sorted_data)

    plt.plot(confidence_bounds, proportions, marker='+', color='red')
    #plt.xlabel('Confidence Bound')
    #plt.ylabel('Proportion of Simulations with Polarization')
    #plt.title('Proportion of Simulations Ending with Polarization by Confidence Bound')
    plt.grid(True)
    #plt.show()

def plot_fragmentation(fragmentation_proportion):
    # Sorting the data by confidence bound for plotting
    sorted_data = sorted(fragmentation_proportion.items())
    confidence_bounds, proportions = zip(*sorted_data)

    plt.plot(confidence_bounds, proportions, marker='d', color='green')
    plt.xlabel('Confidence Bound')
    plt.ylabel('Proportion of Simulations')
    #plt.title('Proportion of Simulations by Opinion Clusters')
    plt.grid(True)
    plt.savefig(f'/Volumes/LaCie/HK_Base/Giuste/Analisi/HK_T/Opinion_Clusters/Confidence_Bound/Plot/{gr}_Clusters_HKT.png')
    plt.close()

graphs = ['CycleR_0.1', 'CycleR_0.2', 'CycleR_0.3', 'ER_0.2', 'ER_0.8']

for gr in graphs: 
    bas_graph = gr.split('_')[-2]
    #print(bas_graph)
    if bas_graph == 'ER':
        #print('yes')
        bas_graph = 'Erdos_Renyi'
    base_dir = f'/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/{bas_graph}/{gr}'
    graph = base_dir.split('/')[-1]
    graph = graph.split('_')[0]
    #print(graph)
    consensus_proportion = aggregate_consensus_data(base_dir)
    plot_consensus(consensus_proportion)
    #perfect_consensus_proportion = aggregate_perfect_consensus_data(base_dir)
    #plot_perfect_consensus(perfect_consensus_proportion)
    polarization_proportion = aggregate_polarization_data(base_dir)
    plot_polarization(polarization_proportion)
    fragmentation_proportion = aggregate_fragmentation_data(base_dir)
    plot_fragmentation(fragmentation_proportion)