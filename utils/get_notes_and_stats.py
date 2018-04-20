#!/usr/bin/env python

import csv
import numpy as np
import os
import sys

# stat234-project
root = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(root)
root = os.path.dirname(os.getcwd())
os.chdir(root)

results_folder = os.path.join(root, 'results' if len(sys.argv) <= 1 else sys.argv[1])
header = None
final_lines = []

def stat_all(root):
    global header
    rd = {}
    # rd['durations'] = {}
    rd['rewards'] = {}
    # rd['losses'] = {}
    # rd['sample_Q'] = {}
    keys = []
    values = []
    print('ROOT IS', root)
    for file in os.listdir(root):  
        path = os.path.join(root, file)              
        if os.path.isdir(path):
            stat_all(path)
        if file.endswith('.csv') and 'clean' not in file and 'notes_and_data' not in file:
            # print('Getting stats of ' + file)
            try:
                reader = csv.reader(open(path, 'r'))
            except FileNotFoundError:
                print(path + ' not found')
                continue
            xs = []
            for r in reader:
                xs.append(float(r[0]))
            if len(xs) == 0:
                return
            xs = xs[int(len(xs) / 2):]
            for key in rd:
                if key in file:
                    rd[key]['mean'] = round(np.mean(xs), 2)
                    rd[key]['std'] = round(np.std(xs), 2)
                    rd[key]['max'] = round(np.max(xs), 2)
                    rd[key]['min'] = round(np.min(xs), 2)
                    rd[key]['quantiles'] = [round(x, 2) for x in np.percentile(xs, [25, 50, 75])]
        if 'notes' in file and file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                for line in f:
                    if ':' in line:
                        pair = line[:-1].split(':')
                        keys.append(pair[0].strip())
                        values.append(pair[1].strip().replace('~ ', '-').replace('~', '-'))

    notes_file = [x for x in os.listdir(root) if 'notes' in x and x.endswith('.txt')]
    if len(notes_file) == 0:
        return
    data = []
    for key in rd: # only key will be duration
        data += [os.path.basename(os.path.normpath(root)), rd[key]['mean'], rd[key]['std'], rd[key]['max'], rd[key]['min']] + rd[key]['quantiles']
    data += values
    data = [str(datum) for datum in data]
    final_lines.append(','.join(data)+'\n')
    header = 'key,mean,std,max,min,25,50,75,' + ','.join(keys) + '\n'
    
stat_all(results_folder)

with open(os.path.join(results_folder, 'notes_and_data.csv'), 'w') as f:
    f.write(header)
    for line in final_lines:
        f.write(line)
