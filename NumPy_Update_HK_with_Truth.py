#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:22:21 2024

@author: filipporiscicalizzio
"""

import numpy as np
import copy
import networkx as nx
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy
#from scipy.sparse import csr_array,coo_array
import os 

#NumPy Version HK with Truth Parameters 

def hk_with_truth(m,t,d_c1,d_t,A,n):
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
  no_influence = np.where(num_neighbors==0)[0]
  self_opinion = np.zeros(n)
  self_opinion[no_influence] = 1

  Anew += np.diag(self_opinion)
  num_neighbors[no_influence] = 1

  Anew = np.transpose(((1-d_t)/num_neighbors)*np.transpose(Anew))

  d_u = np.matmul(Anew,d_c1) + d_t*t

  return d_u

def sim(m,t,d_c,d_t,G,itr):
  #m: confidence parameter
  #t: truth value
  #d_c: dictionary of opinions
  #d_t: dictionary of truth attraction
  #G: network
  #itr: number of maximal iteration of a simulation

  A = nx.to_numpy_array(G)
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
    d_c = hk_with_truth(m,t,d_c,d_t,A,n)
    opinions[count] = d_c
    state_change = np.sum(np.abs(d_c - d_c_previous))
    steady_state_counter = steady_state_counter+1 if state_change<0.00001 else 0
    steady_state_marker = 1 if steady_state_counter >= 100 else 0

  if steady_state_marker == 1:
    convergence_time = count - 100

  df_t = pd.DataFrame(opinions[:count],index=np.arange(count)+1)

  df_t.to_csv(f'{path}/simulation_{i}_{t}_{ta}_{m}.csv', index=False)

  list_plot = [df_t, convergence_time]

  return list_plot

population = [10, 50, 100, 200, 400, 600, 800, 1000]
truth = [0.1, 0.3, 0.5, 0.7, 0.9]
truth_a = [0.1, 0.3, 0.5, 0.7, 0.9]
cb = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
prob_e = ['0.1', '0.2', '0.3']

mc_sim = 20
itr = 1000

#for prop in prob_e:
directory = f'Simulations/Erdos_Renyi/ER_0.1'
for pop in population:
    n = pop
    new_path = f'ER_{pop}_0.1'
    path = os.path.join(directory, new_path)

    if not os.path.exists(path): 
        os.mkdir(path)
    for prim in range(mc_sim):
        G = nx.erdos_renyi_graph(n, float(prop))
        for i in range(mc_sim):
            d_c = np.random.rand(n)
            ts = np.zeros(n)
            ts[:n//2] = 1
            np.random.shuffle(ts)
            for t in truth:
                data = pd.DataFrame()
                data['m'] = cb
                data.set_index('m')
                for ta in truth_a:
                    d_t = ta*ts
                    column = []
                    for m in cb: 
                        dff_new,ct_new = sim(m,t,d_c,d_t,G,itr)
                        column.append(ct_new)
                    data[ta] = column
                data.to_csv(f'{path}/Convergence_Time_{prim}_{i}_{t}_{ta}_{m}.csv')
                
print('THE END')  