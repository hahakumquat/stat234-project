#!/usr/bin/env python

import csv
import os
import sys

root = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(root)
root = os.path.dirname(os.getcwd())
os.chdir(root)

res = []

with open(os.path.join(root, sys.argv[1]), 'r') as f:
    tmp = []
    for line in f:
        if line == ' \n':
            res.append(tmp)
            tmp = []
        else:
            key_val = line[:-1].split(':')
            if len(key_val) == 1:
                tmp.append(key_val[0].split('/')[0])
            elif 'this is a' in key_val:
                tmp.append(-1)
            else:
                tmp.append(key_val[1].strip())


with open(sys.argv[1] + '.csv', 'w') as fw:
    writer = csv.writer(fw, delimiter=',')
    writer.writerow('game,model,target_update,learning_rate,anneal,loss_function,weight_decay'.split(','))
    for r in res:
        writer.writerow(r)

