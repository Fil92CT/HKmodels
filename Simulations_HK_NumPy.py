#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:18:23 2024

@author: filipporiscicalizzio
"""

import numpy as np
import copy
import networkx as nx
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy
import random
#from scipy.sparse import csr_array,coo_array
import os 

#SIMULATIONS HK. PLEASE NOTE THAT FUNCTION HK_WITH_TRUTH *DOES NOT* CONTAIN TRUTH PARAMETERS
#THIS IS AN ADAPTATION FROM THE CODE FOR THE SIMULATIONS FOR HK WITH TRUTH PARAMETERS
#SO SOME NAMES ARE LEGACY FROM THE OLDER SCRIPT

def hk_with_truth(m,d_c1,A,n):
  #m: confidence parameter
  #t: truth value
  #d_c1: copy of dictionary of opinions
  #d_t: dictionary of truth attraction
  #A: adjacency matrix
  #n: number of nodes

  dtemp = np.outer(np.ones(n),d_c1)
  dists = np.abs(dtemp-dtemp.transpose())
  neigh = dists<m
  Anew = neigh*A

  num_neighbors = np.sum(Anew,axis=1)

  #FOLLOWING NEEDED WHEN NO SELF-LOOP
    
  #no_influence = np.where(num_neighbors==0)[0]
  #self_opinion = np.zeros(n)
  #self_opinion[no_influence] = 1

  #Anew += np.diag(self_opinion)
  #num_neighbors[no_influence] = 1

  Anew = np.transpose((1/num_neighbors)*np.transpose(Anew))

  d_u = np.matmul(Anew,d_c1)

  return d_u

def sim(m,d_c,G,itr):
  #m: confidence parameter
  #t: truth value
  #d_c: dictionary of opinions
  #d_t: dictionary of truth attraction
  #G: network
  #itr: number of maximal iteration of a simulation

  A = nx.to_numpy_array(G)
  np.fill_diagonal(A, 1, wrap=False)
  n = len(G)

  count = 0
  opinions = np.empty((itr+1,n))
  opinions[0] = d_c
  steady_state_counter = 0
  steady_state_marker = 0
  convergence_time = itr

  while (count < itr) and (steady_state_marker != 1):
    count += 1
    d_c_previous = np.copy(d_c)
    d_c = hk_with_truth(m,d_c,A,n)
    opinions[count] = d_c
    state_change = np.sum(np.abs(d_c - d_c_previous))
    steady_state_counter = steady_state_counter+1 if state_change<0.00001 else 0
    steady_state_marker = 1 if steady_state_counter >= 100 else 0

  if steady_state_marker == 1:
    convergence_time = count - 100

  df_t = pd.DataFrame(opinions[:count],index=np.arange(count)+1)

  df_t.to_csv(f'{path}/simulation_{i}_{m}.csv', index=False)

  list_plot = [df_t, convergence_time]

  return list_plot

population = [10, 50, 100, 200, 400, 600, 800, 1000]
cb = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#prb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
str_cb = ['0.02', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'] 
mc_sim = 20
itr = 1000

#for pop in population:
    #path = f'Simulations/HK_Base/Cycle/Cycle_{pop}'
    #n = pop
    #G = nx.cycle_graph(n)
    #for i in range(mc_sim):
        #d_c = np.random.rand(n)
        #data = pd.DataFrame(columns=['Confidence Bound'], index= str_cb)
        #data['m'] = cb
        #data.set_index('m')
        #column = []
        #for m in cb: 
            #dff_new,ct_new = sim(m,d_c,G,itr)
            #column.append(ct_new)
            #data = pd.Series(column)
        #data.to_csv(f'{path}/Convergence_Time_{i}_{m}.csv')

#for pr in prb:
for pop in population:
    #n = pop
    G = nx.erdos_renyi_graph(pop, 0.9)
    
    #FOLLOWING 8 LINES ONLY FOR CYCLE WITH RANDOM EDGES
    #num_new_edge = pop * 0.2 
    #Bool = False
    #new_edge = 0
    #while new_edge < num_new_edge: 
        #pair = random.choices(list(G.nodes), k=2)
        #if pair not in G.edges: 
            #G.add_edge(pair[0], pair[1])
            #new_edge += 1
            
    directory = 'Simulations/HK_Base/ER/ER_0.9'
    new_folder = f'ER_{pop}'
    path = os.path.join(directory, new_folder)
    if not os.path.exists(path): 
        os.mkdir(path)
    for i in range(mc_sim):
        d_c = np.random.rand(pop)
        convergence_times = {}  # Dictionary to store convergence time for each cb value
        for m in cb:
            _, ct_new = sim(m, d_c, G, itr)
            convergence_times[f'cb_{m}'] = ct_new
        df = pd.DataFrame([convergence_times])
        df.to_csv(f'{path}/Convergence_Time_{i}.csv', index=False)

                
print('THE END') 