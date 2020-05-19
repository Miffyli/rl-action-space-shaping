#!/bin/bash
# Run obstacle_tower experiments 

timesteps=10000000
repetitions=3
additional_parameters="--subprocenv --num-envs 32 --n-steps 128 --ent-coef 0.001 --cnn"

button_sets="Minimal Full Backward Strafe AlwaysForward"
action_spaces="Discrete MultiDiscrete"

for repetition in $(seq 1 ${repetitions})
do
    for button_set in $button_sets
    do
        for action_space in $action_spaces
        do
            env="ObstacleTower-${button_set}-${action_space}-v0"
            ./scripts/run_stable_baselines.sh ${env} --env ${env} --timesteps ${timesteps} ${additional_parameters} &
            ot_pid=$!

            # Dirty hack: Above script gets
            # stuck somewhere. Wait for it to write
            # a file to tell it is done training, and
            # then we kill it.
            while :
            do
                if [[ -f "_done_training" ]]
                then
                    # Kill process and clean up
                    kill $ot_pid
                    # There is still a strangler there...
                    pkill -f "python3 run_stable_baselines"
                    killall -9 obstacletower.x86_64
                    rm _done_training
                    break
                fi
                sleep 1.0
            done
                
            # Sleep a generous amount to free the ports
            sleep 60.0
        done
    done
done

