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

results_folder = os.path.join(root, 'data' if len(sys.argv) <= 1 else sys.argv[1])
header = None
final_lines = []

def rename_all(root):
    global header
    # print('ROOT IS', root)
    for file in os.listdir(root): 
        path = os.path.join(root, file)               
        if os.path.isdir(path):
            rename_all(path)
        if file.endswith('.pdf'):
            new_file = file.replace('DQN_GS', 'DQN-GS').replace('DQN_PCA', 'DQN-PCA').replace('PCA_Mini', 'PCA-Mini').replace('DQCNN_PCA', 'DQCNN-PCA')
            new_path = os.path.join(root, new_file)
            os.rename(path, new_path)
        if '_mini' in file:
            # print('Getting stats of ' + file)
            print(file)
            
            os.rename(path, path.replace('_mini', '_Mini'))
    
rename_all(results_folder)
