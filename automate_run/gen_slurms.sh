#!/bin/bash

ntrains=(10000)
target_update=(0 100 1000)
weight_decay=(0.001 0.1)
batch_sizes=(128)
losses=(Huber MSE)
anneals=(--noanneal --anneal)
lrs=(0.001 0.0001)
models=(DQN_GS DDQN_GS)
games=(CartPole-v0 Acrobot-v1 MountainCar-v0)

for e in "${ntrains[@]}"
do
    for targ in "${target_update[@]}"
    do
        for reg in "${weight_decay[@]}"
        do
            for batch in "${batch_sizes[@]}"
            do
                for loss in "${losses[@]}"
                do
                    for anl in "${anneals[@]}"
                    do
                        for lr in "${lrs[@]}"
                        do
                            for m in "${models[@]}"
                            do
                                for g in "${games[@]}"
                                do
                                    ./replace.sh $g $m $e $targ $lr $batch $anl $loss $reg
                                    cat slurm.template > tmp.slurm
                                    sed -i s/_name_/$g$m$e$targ$lr$batch$anl$loss$reg/ tmp.slurm
                                    cat tmp.txt >> tmp.slurm
                                    sbatch tmp.slurm
                                done
                            done
                        done
                    done    
                done
            done
        done
    done
done

rm tmp*
