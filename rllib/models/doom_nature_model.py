import gym
import sonnet as snt
import tensorflow as tf
import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


def build_logits(action_space, latent_vec):
    if isinstance(action_space, gym.spaces.Discrete):
        return snt.Linear(output_size=action_space.n,
                          initializers={'w':  tf.initializers.orthogonal(np.sqrt(0.01))})(latent_vec)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return tf.concat([build_logits(gym.spaces.Discrete(n), latent_vec) for n in action_space.nvec], axis=1)
    elif isinstance(action_space, gym.spaces.Box):
        assert len(action_space.shape) == 1
        mean = snt.Linear(output_size=action_space.shape[0],
                          initializers={'w':  tf.initializers.orthogonal(np.sqrt(0.01))})(latent_vec)
        log_std = tf.get_variable(name='log_std', shape=[1, action_space.shape[0]], initializer=tf.zeros_initializer())
        return tf.concat([mean, mean * 0.0 + log_std], axis=1)
    elif isinstance(action_space, gym.spaces.Tuple):
        return tf.concat([build_logits(space, latent_vec) for space in action_space.spaces], axis=1)
    else:
        raise NotImplementedError(f"Action space of type {type(action_space)} is not supported.")


class NatureModel(snt.AbstractModule):

    def __init__(self, action_space, _sentinel=None, custom_getter=None, name=None):
        super().__init__(_sentinel, custom_getter, name)
        self.action_space = action_space

    def _build(self, spatial_obs, non_spatial_obs=None, *unused_args, **unused_kwargs):
        conv_out = snt.Conv2D(output_channels=32, kernel_shape=8, stride=4,
                              initializers={'w': tf.initializers.orthogonal(np.sqrt(2))})(spatial_obs)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.Conv2D(output_channels=64, kernel_shape=4, stride=2,
                              initializers={'w': tf.initializers.orthogonal(np.sqrt(2))})(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.Conv2D(output_channels=64, kernel_shape=3, stride=1,
                              initializers={'w': tf.initializers.orthogonal(np.sqrt(2))})(conv_out)
        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)
        conv_out = snt.Linear(output_size=512, initializers={'w': tf.initializers.orthogonal(np.sqrt(2))})(conv_out)
        conv_out = tf.nn.relu(conv_out)

        if non_spatial_obs is not None:
            conv_out = tf.concat([conv_out, non_spatial_obs], axis=-1)

        logits = build_logits(action_space=self.action_space, latent_vec=conv_out)
        baseline = snt.Linear(output_size=1,
                              initializers={'w':  tf.initializers.orthogonal(np.sqrt(0.01))})(conv_out)
        return logits, baseline


class DoomNatureModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = NatureModel(action_space)

    def forward(self, input_dict, state, seq_lens):
        obs, prev_actions = input_dict['obs'], input_dict["prev_actions"]

        if isinstance(obs, tuple) or isinstance(obs, list):
            logits, baseline = self.model(obs[0], non_spatial_obs=obs[1])
        else:
            logits, baseline = self.model(obs)

        self.baseline = tf.reshape(baseline, [-1])

        return logits, state

    def variables(self, as_dict=False):
        if not self.model.is_connected:
            var_list = []
        else:
            var_list = self.model.variables
        if as_dict:
            return {v.name: v for v in var_list}
        return var_list

    def value_function(self):
        return self.baseline
