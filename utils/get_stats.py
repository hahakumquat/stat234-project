import csv
import numpy as np
import os

# stat234project
root = os.path.dirname(os.getcwd())

results_folder = os.path.join(root, 'results')

def stat_all(root):
    rd = {}
    rd["durations"] = {}
    rd["rewards"] = {}
    rd["losses"] = {}
    rd["sample_Q"] = {}
    print("ROOT IS", root)
    for file in os.listdir(root):                
        if os.path.isdir(os.path.join(root, file)):
            stat_all(os.path.join(root, file))
        if file.endswith('.csv'):
            print("Getting stats of " + file)
            path = os.path.join(root, file)
            try:
                reader = csv.reader(open(path, 'r'))
            except FileNotFoundError:
                print(path + ' not found')
                continue
            xs = []
            for r in reader:
                xs.append(float(r[0]))

            for key in rd:
                if key in file:
                    rd[key]["mean"] = round(np.mean(xs), 2)
                    rd[key]["std"] = round(np.std(xs), 2)
                    rd[key]["max"] = round(np.max(xs), 2)
                    rd[key]["min"] = round(np.min(xs), 2)
                    rd[key]["quantiles"] = [round(x, 2) for x in np.percentile(xs, [25, 50, 75])]

    notes_file = [x for x in os.listdir(root) if 'notes' in x]
    if len(notes_file) == 0:
        return
    notes_file = notes_file[0].replace('notes', 'stats')
    with open(os.path.join(root, notes_file), 'w') as f:
        f.write('key,mean,std,max,min,25,50,75\n')
        for key in rd:
            data = [key, str(rd[key]["mean"]), str(rd[key]["std"]), str(rd[key]["max"]), str(rd[key]["min"]), str(rd[key]["quantiles"][0]), str(rd[key]['quantiles'][1]), str(rd[key]['quantiles'][2])]
            f.write(','.join(data)+'\n')
        
stat_all(results_folder)
