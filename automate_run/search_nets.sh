#!/bin/bash

games=(CartPole-v0 Acrobot-v1 MountainCar-v0)
params=(16 16_32 16_32_64 64_32_16 64_16 64 32 32_32_16)
for g in "${games[@]}"
do
    for p in "${params[@]}"
    ./replace.sh $g DQN_PCA 10000 0 0.001 128 --anneal Huber 0.1 --net_params p
    cat slurm.template > tmp2.slurm
    sed -i s/_name_/$gDQN_PCA00.001128--annealHuber0.1$p/ tmp2.slurm
    cat tmp.txt >> tmp2.slurm
    sbatch tmp2.slurm
done
