import matplotlib.pyplot as plt
import csv
import os
script_dir = os.path.dirname(os.getcwd())

data = ['cartpole/durations.csv', 'cartpole/rewards.csv', 
        'acrobot/durations.csv', 'acrobot/rewards.csv', 
        'mountaincar/durations.csv', 'mountaincar/rewards.csv', ]

for d in data:
    path = os.path.join(script_dir, d)
    try:
    	reader = csv.reader(open(path, 'r'))
    except FileNotFoundError:
    	print(path + ' not found')
    	continue
    xs = [r[0] for r in reader]
    plt.plot(xs)
    plt.title(os.path.basename(path).split('.')[0])
    plt.xlabel('episodes')
    end_dir = os.path.join(script_dir, os.path.dirname(d), os.path.basename(path).split('.')[0] + '.pdf')
    plt.savefig(end_dir)
    plt.close()
