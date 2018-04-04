#!/bin/bash

cd ../results/
for game in CartPole-v0 Acrobot-v1 MountainCar-v0
do
    cd $game
    for result in $(ls)
    do
        cd $result
        cat $(ls | grep "notes")
        plots=$(find . -name "*.pdf")
        evince -f $plots
        cd ..
        echo " "
    done
    cd ..
done

