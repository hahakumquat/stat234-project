import matplotlib.pyplot as plt
import csv
import os
script_dir = os.path.dirname(os.getcwd())

data = ['cartpole', 
        'acrobot', 
        'mountaincar', ]

for directory in data:
    absolute_directory = os.path.join(script_dir, directory)
    for file in os.listdir(absolute_directory):
        if file.endswith('.csv') and 'losses' not in file:
            path = os.path.join(absolute_directory, file)
            print(path)
            try:
                reader = csv.reader(open(path, 'r'))
            except FileNotFoundError:
                print(path + ' not found')
                continue
            xs = [float(r[0]) for r in reader]
            plt.plot(xs)
            plt.title(directory + ' ' + file.split('.')[0])
            plt.xlabel('episodes')
            end_dir = path.split('.')[0] + '.pdf'
            plt.savefig(end_dir)
            plt.close()
