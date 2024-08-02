#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:52:45 2024

@author: filipporiscicalizzio
"""

import os
import pandas as pd
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize


graph_list = ['ER_0.1', 'ER_0.3', 'ER_0.9']

for gr in graph_list:
    
    # Define your parameters
    pop = [10, 50, 100, 200, 400, 600, 800, 1000]
    ta = ['0.1', '0.3', '0.5', '0.7', '0.9']
    cb = ['0.02', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    
    directory = f'/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_T/Erdos_Renyi/{gr}'
    graph = directory.split('/')[-1]
    #print(graph)
    graph_n = graph.split('_')[0]
    #graph_num = graph.split('_')[1]
    #name = graph_n + graph_num
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(pop), figsize=(20, 10), sharey=True)
    fig.subplots_adjust(bottom=0.2)  # Adjust the bottom to have space for the color bar
    
    # Storage for the global min and max values
    global_min, global_max = np.inf, -np.inf
    
    for idx, p in enumerate(pop): 
        # Path to the directory containing your CSV files
        folder_path = f'{directory}/{graph_n}_{p}' # Update with your actual folder path
        #print(folder_path)
        # Initialize heatmap_data with NaNs
        heatmap_data = pd.DataFrame(index=cb, columns=ta, data=np.nan)
    
        # Collect all data for each cb_value and ta_value
        for cb_value in cb:
            for ta_value in ta:
                all_convergence_times = []
        
                # Iterate over files in the folder
                for file in os.listdir(folder_path):
                    #print(file)
                    if file.startswith('Convergence'):
                        #print('YES')
                        # Construct the full file path
                        file_path = os.path.join(folder_path, file)
        
                        # Read the DataFrame from the file
                        df = pd.read_csv(file_path)
                        df.drop('Unnamed: 0', axis=1, inplace=True)
                        
                        # Set 'm' as the index
                        df.set_index('m', inplace=True)
                        #print(df)
        
                        # Extract the convergence times for this cb_value and ta_value
                        convergence_times = df.at[float(cb_value), ta_value]
                        
                        # Append to the list of all convergence times
                        all_convergence_times.append(convergence_times)
                        #print(all_convergence_times)
        
                # Calculate the mean of all convergence times
                if all_convergence_times:
                    mean_convergence_time = np.mean(all_convergence_times)
                    heatmap_data.at[cb_value, ta_value] = mean_convergence_time
                    #print(heatmap_data)
                # Compute global min and max
                global_min = heatmap_data.min().min()
                global_max = heatmap_data.max().max()
        
        # Create a Normalize object with your min and max values
        norm = LogNorm(vmin=global_min, vmax=global_max)
        # Plot the heatmap
        #print(heatmap_data)
        sns.heatmap(heatmap_data.astype(float), ax=axes[idx], cmap='viridis', norm=LogNorm(), annot=False)
        axes[idx].set_title(f'{p}')
        axes[idx].set_xlabel('Truth Attraction')
        if idx == 0:
            axes[idx].set_ylabel('Confidence Bound')
    
    
    # Hide color bars for all heatmaps
    for ax in axes:
        ax.collections[0].colorbar.remove()
    
    # Add a single color bar at the bottom of the figure
    cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])  # x-position, y-position, width, height
    #norm = LogNorm(vmin=global_min, vmax=global_max)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_xlabel('Mean Convergence Time')
    
    # Save the figure
    plt.show()
    plt.savefig(f'/Volumes/LaCie/HK_Base/Giuste/Analisi/HK_T/Convergence_Time/Heatmaps/ER/Heatmaps_{gr}_Mean.png')
    #plt.close()