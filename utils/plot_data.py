import matplotlib.pyplot as plt
import csv
import os
import sys

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)

script_dir = os.path.dirname(os.getcwd())

directory = 'data' if len(sys.argv) > 1 and sys.argv[1] == 'data' else 'results'

root = os.path.join(script_dir, directory)

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
            xs = [float(r[0]) for r in reader]
            plt.plot(xs)
            plt.title(os.path.basename(root) + ' ' + file.split('_')[-1][:-4])
            plt.xlabel('episodes')
            end_dir = path.split('.')[0] + '.pdf'
            plt.savefig(end_dir)
            plt.close()

plot_all(root)
