#!/bin/bash
# Run experiment with stable-baselines

if test -z "$1"
then
    echo "Usage: run_stable_baselines experiment_name [parameters_to_train ...]"
    exit
fi

experiment_dir=experiments/stable_baselines_${1}_$(date -Iseconds)
# Prepare directory
./scripts/prepare_experiment_dir.sh ${experiment_dir}
# Store the launch parameters there
echo ${@:0} > ${experiment_dir}/launch_arguments.txt

# Run code
python3 run_stable_baselines.py \
  --output=${experiment_dir} \
  ${@:2} \
  | tee ${experiment_dir}/stdout.txt
