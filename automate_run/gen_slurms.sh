#!/bin/bash

ntrains=(100000)
target_update=(0)
weight_decay=(0.1)
batch_sizes=(128)
losses=(Huber)
anneals=(--anneal)
lrs=(0.001)
models=(DQN_GS DQN_PCA DDQN_GS DDQCNN_PCA DQCNN_PCA DQCNN_PCA_Mini DDQCNN_PCA_mini)
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
                                    cat slurm.template > tmp2.slurm
                                    sed -i s/_name_/$g$m$e$targ$lr$batch$anl$loss$reg/ tmp2.slurm
                                    cat tmp.txt >> tmp2.slurm
                                    sbatch tmp2.slurm
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
