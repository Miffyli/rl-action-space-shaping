#!/bin/bash
# Run experiments with vizdoom hgs experiments

./scripts/run_vizdoom_get_to_goal_discrete1.sh &
./scripts/run_vizdoom_get_to_goal_discrete2.sh &
./scripts/run_vizdoom_get_to_goal_discreteall.sh &
./scripts/run_vizdoom_get_to_goal_multidiscrete.sh &
wait $!
./scripts/run_vizdoom_hgs_discrete1.sh &
./scripts/run_vizdoom_hgs_discrete2.sh &
./scripts/run_vizdoom_hgs_discreteall.sh &
./scripts/run_vizdoom_hgs_multidiscrete.sh &
wait $!
./scripts/run_vizdoom_deathmatch_discrete1.sh &
./scripts/run_vizdoom_deathmatch_discrete2.sh &
./scripts/run_vizdoom_deathmatch_discreteall.sh &
./scripts/run_vizdoom_deathmatch_multidiscrete.sh &
