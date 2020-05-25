#!/bin/bash
# Run basic stable-baselines + GetToGoal experiments

timesteps=500000
repetitions=10
button_sets="Minimal BareMinimum Backward Strafe"


# Standard experiment
for button_set in ${button_sets}
do
  env=ViZDoom-GetToGoal-Discrete1-${button_set}-v0
  for repetition in $(seq 1 ${repetitions})
  do
    ./scripts/run_stable_baselines.sh ${env} --env ${env} --timesteps ${timesteps} --cnn
  done
done

