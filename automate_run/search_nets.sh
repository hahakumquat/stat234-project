#!/bin/bash

games=(CartPole-v0 Acrobot-v1 MountainCar-v0)
params=(128 64 128_64 128_64_32 128_128_64_32)
models=(DQN_PCA DDQN_PCA)
for g in "${games[@]}"
do
    for p in "${params[@]}"
    do
	for m in "${models[@]}"       
	do
	    ./replace.sh $g $m 20000 0 0.001 128 "--anneal" Huber 0.1 $p
            cat slurm.template > tmp2.slurm
            sed -i s/_name_/$g$m00.001128--annealHuber0.1$p/ tmp2.slurm
            cat tmp.txt >> tmp2.slurm
            sbatch tmp2.slurm
	done
    done
done
