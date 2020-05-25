#!/bin/bash
# Run basic stable-baselines + GetToGoal experiments

timesteps=10000000
repetitions=5
additional_params="--atari-ppo --cnn"

env="MsPacmanNoFrameskip-v4"

for repetition in $(seq 1 ${repetitions})
do
    # Standard run, no modifications
    ./scripts/run_stable_baselines.sh ${env}-Minimal --env ${env} --timesteps ${timesteps} ${additional_params}
    # Full actions
    ./scripts/run_stable_baselines.sh ${env}-Full --env ${env} --timesteps ${timesteps} --atari-full-actions ${additional_params}
    # Multi-discrete
    ./scripts/run_stable_baselines.sh ${env}-MultiDiscrete --env ${env} --timesteps ${timesteps} --atari-full-actions --atari-multidiscrete ${additional_params}
done

