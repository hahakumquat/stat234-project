import csv
import numpy as np
import os
import sys

# stat234-project
root = os.path.dirname(os.getcwd())

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
        if 'notes_cpu_.txt' in file:
            # print('Getting stats of ' + file)
            print(file)
            
            os.rename(path, path.replace('notes_cpu_.txt', 'notes_cpu.txt'))
    
rename_all(results_folder)
