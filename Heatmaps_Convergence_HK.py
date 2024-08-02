#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:31:32 2023

@author: filipporiscicalizzio
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors


"""CODE HEATMAPS CONVERGENCE TIME HK"""

def read_and_average_data(folder_path):
    """Read all CSV files starting with 'Convergence_Time_' in the given folder and return the average values for each column."""
    file_names = [f for f in os.listdir(folder_path) if f.startswith("Convergence_Time_") and f.endswith(".csv")]
    all_data = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        data = pd.read_csv(file_path)
        all_data.append(data)

    return pd.concat(all_data).mean()

def main():
    base_folder = "/Volumes/LaCie/HK_Base/Giuste/Simulazioni/HK_Base/Cycle"  # Adjust the path if necessary
    heatmap_data = []  # Ensure this is a list

    for folder_name in os.listdir(base_folder):
        if folder_name.startswith("Cycle_"):
            folder_path = os.path.join(base_folder, folder_name)
            if os.path.isdir(folder_path):
                # Extract the number of nodes from the folder name
                num_nodes = int(folder_name.split('_')[1])
                avg_data = read_and_average_data(folder_path)

                for col in avg_data.index:
                    if col.startswith('cb_'):
                        cb_value = float(col.split('_')[1])
                        heatmap_data.append({'Population': num_nodes, 'cb': cb_value, 'MeanTime': avg_data[col]})

    # Preparing data for the heatmap
    df = pd.DataFrame(heatmap_data)
    df_pivot = df.pivot(index='cb', columns='Population', values='MeanTime')

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, annot=False, cmap="viridis", norm=matplotlib.colors.LogNorm())
    plt.title("Heatmap of Mean Convergence Time. Cycle")
    plt.xlabel("Population")
    plt.ylabel("Confidence Bound")
    plt.savefig('/Volumes/LaCie/HK_Base/Giuste/Analisi/HK_Base/Convergence_Time/Heatmaps/Cycle/HEATMAP.png')
    plt.show()

if __name__ == "__main__":
    main()
