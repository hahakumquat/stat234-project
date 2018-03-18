import matplotlib.pyplot as plt
import csv
import os
script_dir = os.path.dirname(os.getcwd())

data = ['cartpole/durations_DQN.csv', 'cartpole/rewards.csv']

for d in data:
    path = os.path.join(script_dir, d)
    try:
        reader = csv.reader(open(path, 'r'))
        xs = [r[0] for r in reader]
        plt.plot(xs)
        end_dir = os.path.join(script_dir, os.path.dirname(d), os.path.basename(path).split('.')[0] + '.pdf')
        plt.savefig(end_dir)
    except:
        continue
