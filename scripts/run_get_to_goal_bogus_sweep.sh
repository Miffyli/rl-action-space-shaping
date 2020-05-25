#!/bin/bash
# Try out how number of bogus ("noop") actions
# affects learning

timesteps=2000000
repetitions=10
net_arch="64 64"

# Sweep over number of bogus actions
# These are done via the environment names (yay),
# so we need to construct appropiate env string
for env_prefix in "GetToGoal-Discrete" "GetToGoal-MultiDiscrete" "GetToGoal-TankDiscrete" "GetToGoal-TankMultiDiscrete" "GetToGoal-Continuous"
do
  for num_bogus in $(seq 0 20)
  do
    for repetition in $(seq 1 ${repetitions})
    do
        env=${env_prefix}-Bogus${num_bogus}-v0
        CUDA_VISIBLE_DEVICES="" ./scripts/run_stable_baselines.sh ${env} --env ${env} --timesteps ${timesteps} --net-arch ${net_arch} &
        # Sleep a little to interleave the runs (and to get right timestamp, derp)
        sleep 1.1s
    done
    # Wait for the last one to finish
    wait $!
  done
done
