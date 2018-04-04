#!/bin/bash

cd ../results/
for game in CartPole-v0 Acrobot-v1 MountainCar-v0
do
    cd $game
    for result in $(ls)
    do
        echo $game/$result
        cd $result
        cat $(ls | grep "notes")
        plots=$(find . -name "*.pdf")
        evince -f $plots 
        # evince -f $plots & echo "[g]ood or [b]ad?" & read res
        # if [ $res = "b" ]
        # then
        #     echo "Deleting PDFs."
        #     rm *.pdf
        # else
        #     echo "Continuing."
        # fi
        cd ..
        echo " "
    done
    cd ..
done

