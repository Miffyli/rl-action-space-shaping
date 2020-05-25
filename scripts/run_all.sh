#!/bin/bash
# Run all experiments

./scripts/run_get_to_goal_all.sh
./scripts/run_vizdoom_all.sh
./scripts/run_atari_all.sh
./scripts/run_obstacletower.sh
./scripts/run_rllib_gettogoal.sh
./scripts/run_rllib_vizdoom.sh
