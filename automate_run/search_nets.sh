#!/bin/bash

games=(CartPole-v0 Acrobot-v1 MountainCar-v0)
params=(32_16 64_32 64_64 32_32)
# params=(16 16_32 16_32_64 64_32_16 64_16 64 32 32_32_16 128 128_64 128_64_32 128_128_64_32)
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
