# Action Space Shaping in Deep Reinforcement Learning

Experiment code for the paper "[Action Space Shaping in Deep Reinforcement Learning](https://arxiv.org/abs/2004.00980)".

Testing different ways of modifying action spaces in various environments, and how that affects the learning performance.
Long story short: Removing some "unnecessary" actions can be crucial for learning. Continuous actions should be discretized,
but converting multi-discrete spaces to discrete is not too helpful.

## Requirements
* stable-baselines (v2.10.0)
* rllib
* cv2
* [obstacle-tower-env](https://github.com/Unity-Technologies/obstacle-tower-env)
* [ViZDoom](https://github.com/Marqt/ViZDoom/) (v1.1.7)
* OpenAI Gym with Atari environments
* Pillow

## Running experiments


## Plotting

After running above experiments, run following to create figures shown in paper:

```
mkdir -p figures
python3 plot_paper.py experiments figures
```

If you lack some of the experiment files / plotting crashes in specific game, comment
out corresponding lines in `plot_paper.py` main function `main(args)` (near the end of the file).

