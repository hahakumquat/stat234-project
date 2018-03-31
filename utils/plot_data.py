import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
 
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
    print(len(res))
    return res
 
def plot_all(root):
    for file in os.listdir(root):
        if os.path.isdir(os.path.join(root, file)):
            plot_all(os.path.join(root, file))
        if file.endswith('.csv'):
            print("Plotting " + file)
            path = os.path.join(root, file)
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
            
            plt.title(plot_type)
            plt.xlabel('episodes' if 'losses' not in plot_type else 'number of trains')
            plt.legend()
            end_dir = path.split('.')[0] + '.pdf'
            plt.savefig(end_dir)
            plt.close()
 
plot_all(root)
