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
        plots=$(find . -name "*durations*.pdf")
        evince -f $plots & echo "[g]ood or [b]ad or o[k]?" & read res        
        if [ $res = "b" ]
        then
            echo "Bad file."
            echo $game/$result >> ../../bad.txt
            cat $(ls | grep "notes") >> ../../bad.txt
            echo " " >> ../../bad.txt
        elif [ $res = "k" ]
        then
            echo "Ok file."
            echo $game/$result >> ../../ok.txt
            cat $(ls | grep "notes") >> ../../ok.txt
            echo " " >> ../../ok.txt
        else
            echo "Good file."
            echo $game/$result >> ../../good.txt
            cat $(ls | grep "notes") >> ../../good.txt
            echo " " >> ../../good.txt
        fi
        cd ..
    done
    cd ..
done

