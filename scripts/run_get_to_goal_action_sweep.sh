#!/bin/bash
# Try out how number of Discrete/MultiDiscrete actions
# affects the results

timesteps=2000000
repetitions=10
net_arch="64 64"

# Sweep over actions
# These are done via the environment names (yay),
# so we need to construct appropiate env string
for env_prefix in "GetToGoal-Discrete" "GetToGoal-MultiDiscrete"
do
  # Start from three actions, which is the first where we can actually
  # solve the task
  for num_actions in $(seq 3 50)
  do
    for repetition in $(seq 1 ${repetitions})
    do
        env=${env_prefix}${num_actions}-v0
        CUDA_VISIBLE_DEVICES="" ./scripts/run_stable_baselines.sh ${env} --env ${env} --timesteps ${timesteps} --net-arch ${net_arch} &
        # Sleep a little to interleave the runs (and to get right timestamp, derp)
        sleep 1.1s
    done
    # Wait for the last one to finish
    wait $!
  done
done
