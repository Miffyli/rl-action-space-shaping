# Action Space Shaping in Deep Reinforcement Learning

Experiment code for the paper "[Action Space Shaping in Deep Reinforcement Learning](https://arxiv.org/abs/2004.00980)".

Testing different ways of modifying action spaces in various environments, and how that affects the learning performance.
Long story short: Removing some "unnecessary" actions can be crucial for learning. Continuous actions should be discretized,
but converting multi-discrete spaces to discrete is not too helpful.

Note: This does not include Starcraft 2 experiment code as of yet, as it is part of ongoing research. The IMPALA implementation used for that is
based on [Deepmind scalable-agents](https://github.com/deepmind/scalable_agent) code.

## Requirements

See `requirements.txt` for most requirements. Additionally [obstacle-tower-env](https://github.com/Unity-Technologies/obstacle-tower-env) for running ObstacleTower experiments.

## Running experiments

In root directory, run `./scripts/run_all.sh`. If all goes well, this should create `experiments` directory
with bunch of results. 

## Plotting

After running above experiments, run following to create figures shown in paper:

```
mkdir -p figures
python3 plot_paper.py experiments figures
```

If you lack some of the experiment files / plotting crashes in specific game, comment
out corresponding lines in `plot_paper.py` main function `main(args)` (near the end of the file).

