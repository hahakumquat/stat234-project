#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns
import sys

# sns.set()
 
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
 
script_dir = os.path.dirname(os.getcwd())
 
directory = sys.argv[1] if len(sys.argv) > 1 else 'results'
y_lo = float("-inf")
y_hi = float("inf")
res = 20
if len(sys.argv) >= 3:
    y_lo = float(sys.argv[2])
if len(sys.argv) >= 4:
    y_hi = float(sys.argv[3])
if len(sys.argv) >= 5:
    res = float(sys.argv[4])
    
root = os.path.join(script_dir, directory)
 
def running_mean(lst, k):
    cumsum = np.cumsum(np.insert(lst, 0, 0)) 
    res = (cumsum[k:] - cumsum[:-k]) / float(k)
    return res
 
def plot_all(root):
    for file in os.listdir(root):
        if os.path.isdir(os.path.join(root, file)):
            plot_all(os.path.join(root, file))
        if (file.endswith('.csv')
           and 'clean' not in file
           and 'notes' not in file
           and 'states' not in file):
           # and not os.path.exists(os.path.join(root, file.replace('.csv', '.pdf')))):
            path = os.path.join(root, file)
            print("Plotting " + path)
            try:
                reader = csv.reader(open(path, 'r'))
            except FileNotFoundError:
                print(path + ' not found')
                continue
            xs = np.clip([float(r[0]) for r in reader], y_lo, y_hi)
            plt.plot(xs, label='raw')
            resolution = max(int(len(xs) / res), 1)
            means = running_mean(xs, resolution)
            plt.plot(range(resolution, len(means) + resolution), means, label='rolling mean over ' + str(resolution))

            # title
            try:
                plot_type = file[:file.index('_cpu')]
            except ValueError:
                plot_type = file[:file.index('.csv')]
            
            plt.title(' '.join(plot_type.split('_')))
            plt.xlabel('episodes' if 'losses' not in plot_type else 'number of trains')
            for potential_label in ['reward', 'duration', 'loss', 'sample_Q']:
                if potential_label in plot_type:
                    ylabel = potential_label
            plt.ylabel(ylabel)
            if 'rewards' in plot_type:
                if 'CartPole' in plot_type:
                    plt.ylim([0, 200])
                elif 'Acrobot' in plot_type:
                    plt.ylim([-2000, 0])
                elif 'MountainCar' in plot_type:
                    plt.ylim([-5000, 0])
            elif 'durations' in plot_type:
                if 'CartPole' in plot_type:
                    plt.ylim([0, 200])
                elif 'Acrobot' in plot_type:
                    plt.ylim([0, 2000])
                elif 'MountainCar' in plot_type:
                    plt.ylim([0, 5000])
            plt.legend()
            end_dir = path.split('.')[0] + '.pdf'
            plt.savefig(end_dir)
            plt.close()
 
plot_all(root)
