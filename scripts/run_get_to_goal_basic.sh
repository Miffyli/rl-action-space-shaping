#!/bin/bash
# Run basic stable-baselines + GetToGoal experiments

timesteps=2000000
repetitions=10
net_arch="64 64"

# Standard experiment
for env in "GetToGoal-Discrete-v0" "GetToGoal-MultiDiscrete-v0" "GetToGoal-TankDiscrete-v0" "GetToGoal-TankDiscrete-NoBackward-v0" "GetToGoal-TankDiscrete-Strafe-v0" "GetToGoal-TankMultiDiscrete-v0" "GetToGoal-TankMultiDiscrete-NoBackward-v0" "GetToGoal-TankMultiDiscrete-Strafe-v0" "GetToGoal-Continuous-v0"
do
  for repetition in $(seq 1 ${repetitions})
  do
    CUDA_VISIBLE_DEVICES="" ./scripts/run_stable_baselines.sh ${env} --env ${env} --timesteps ${timesteps} --net-arch ${net_arch} &
    # Sleep a little to interleave the runs (and to get right timestamp, derp)
    sleep 1.1s
  done
  # Wait for the last one to finish
  wait $!
done

