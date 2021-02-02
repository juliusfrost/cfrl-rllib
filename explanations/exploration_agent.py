# import tensorflow as tf 
# import sys
# sys.path.append('../')
# sys.path.append('../dads/unsupervised_skill_learning')
# from dads.unsupervised_skill_learning.dads_agent import DADSAgent
# from dads.unsupervised_skill_learning.skill_dynamics import SkillDynamics

# # policy_dir = '/Users/ericweiner/Documents/cfrl-rllib/dads_policies/policy/ckpt-150.data-00001-of-00002'
# # saved_policy = tf.compat.v2.saved_model.load(policy_dir)
# # s = tf.Session()
# # with tf.Session() as sess:
# #     saver = tf.train.import_meta_graph('/Users/ericweiner/Documents/cfrl-rllib/dads_policies/dynamics/ckpt-150.meta')
# #     saver.restore(sess, tf.train.latest_checkpoint('/Users/ericweiner/Documents/cfrl-rllib/dads_policies/dynamics'))

# # WORKS
# # meta_file = '/Users/ericweiner/Downloads/models/dynamics/ckpt-150.meta'
# # with tf.Session(graph=detection_graph) as sess:
# # sk = SkillDynamics(44, 2)
# # sk.make_placeholders()
# # sk.build_graph()
# # sk.create_saver('/Users/ericweiner/Downloads/models/dynamics')
# # sk.restore_variables()
# # SAC Parameters
# # time_step_spec=tf_agent_time_step_spec,
# #         action_spec=tf_action_spec,
# #         actor_network=actor_net,
# #         critic_network=critic_net,
# #         target_update_tau=0.005,
# #         target_update_period=1,
# #         actor_optimizer=tf.compat.v1.train.AdamOptimizer(
# #             learning_rate=FLAGS.agent_lr),
# #         critic_optimizer=tf.compat.v1.train.AdamOptimizer(
# #             learning_rate=FLAGS.agent_lr),
# #         alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
# #             learning_rate=FLAGS.agent_lr),
# #         td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
# #         gamma=FLAGS.agent_gamma,
# #         reward_scale_factor=1. /
# #         (FLAGS.agent_entropy + 1e-12),
# #         gradient_clipping=None,
# #         debug_summaries=FLAGS.debug,
# #         train_step_counter=global_step)
# da = DADSAgent('/Users/ericweiner/Downloads/models/agent',3)
# # with tf.compat.v1.Session(sk._graph()) as sess:
    

#     # saver = tf.compat.v1.train.import_meta_graph(meta_file)
#     # saver.restore(sess, '/Users/ericweiner/Downloads/models/dynamics/ckpt-150')

# # agent = DADSAgent('/Users/ericweiner/Downloads/models', 10)

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import flags, logging
import functools

import sys
sys.path.append(os.path.abspath('./'))
# import sys
sys.path.append('../')
sys.path.append('../dads/unsupervised_skill_learning')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories.trajectory import from_transition, to_transition
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import policy_step
# from tf_agents.policies import py_tf_policy
# from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

import dads.unsupervised_skill_learning.dads_agent as dads_agent

from dads.envs import skill_wrapper
from dads.envs import video_wrapper
from dads.envs.gym_mujoco import ant
from dads.envs.gym_mujoco import half_cheetah
from dads.envs.gym_mujoco import humanoid
from dads.envs.gym_mujoco import point_mass

from dads.envs import dclaw
from dads.envs import dkitty_redesign
from dads.envs import hand_block

from envs.driving import register, driving_creator
from dads.lib import py_tf_policy
from dads.lib import py_uniform_replay_buffer

# FLAGS = flags.FLAGS
nest = tf.nest

from dads.unsupervised_skill_learning.dads_off import FLAGS

# global variables for this script
observation_omit_size = 0
goal_coord = np.array([10., 10.])
sample_count = 0
iter_count = 0
episode_size_buffer = []
episode_return_buffer = []

# add a flag for state dependent std
def _normal_projection_net(action_spec, init_means_output_factor=0.1):
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)

def print_dict(**kwargs):
  print(kwargs)

#TODO: Delete
def get_environment(env_name='point_mass'):
  global observation_omit_size
  if env_name == 'Ant-v1':
    env = ant.AntEnv(
        expose_all_qpos=True,
        task='motion')
    observation_omit_size = 2
  elif env_name == 'Ant-v1_goal':
    observation_omit_size = 2
    return wrap_env(
        ant.AntEnv(
            task='goal',
            goal=goal_coord,
            expose_all_qpos=True),
        max_episode_steps=FLAGS.max_env_steps)
  elif env_name == 'Ant-v1_foot_sensor':
    env = ant.AntEnv(
        expose_all_qpos=True,
        model_path='ant_footsensor.xml',
        expose_foot_sensors=True)
    observation_omit_size = 2
  elif env_name == 'HalfCheetah-v1':
    env = half_cheetah.HalfCheetahEnv(expose_all_qpos=True, task='motion')
    observation_omit_size = 1
  elif env_name == 'Humanoid-v1':
    env = humanoid.HumanoidEnv(expose_all_qpos=True)
    observation_omit_size = 2
  elif env_name == 'point_mass':
    env = point_mass.PointMassEnv(expose_goal=False, expose_velocity=False)
    observation_omit_size = 2
  elif env_name == 'DClaw':
    env = dclaw.DClawTurnRandom()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DClaw_randomized':
    env = dclaw.DClawTurnRandomDynamics()
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_redesign':
    env = dkitty_redesign.BaseDKittyWalk(
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'DKitty_randomized':
    env = dkitty_redesign.DKittyRandomDynamics(
        randomize_hfield=FLAGS.randomize_hfield,
        expose_last_action=FLAGS.expose_last_action,
        expose_upright=FLAGS.expose_upright,
        robot_noise_ratio=FLAGS.robot_noise_ratio,
        upright_threshold=FLAGS.upright_threshold)
    observation_omit_size = FLAGS.observation_omission_size
  elif env_name == 'HandBlock':
    observation_omit_size = 0
    env = hand_block.HandBlockCustomEnv(
        horizontal_wrist_constraint=FLAGS.horizontal_wrist_constraint,
        vertical_wrist_constraint=FLAGS.vertical_wrist_constraint,
        randomize_initial_position=bool(FLAGS.randomized_initial_distribution),
        randomize_initial_rotation=bool(FLAGS.randomized_initial_distribution))
  elif env_name == 'DrivingPLE-v0':
    observation_omit_size = 0
    env = driving_creator()
  else:
    # note this is already wrapped, no need to wrap again
    env = suite_mujoco.load(env_name)
  return env

def hide_coords(time_step):
  global observation_omit_size
  if observation_omit_size > 0:
    sans_coords = time_step.observation[observation_omit_size:]
    return time_step._replace(observation=sans_coords)

  return time_step


# hard-coding the state-space for dynamics
def process_observation(observation):

  def _shape_based_observation_processing(observation, dim_idx):
    if len(observation.shape) == 1:
      return observation[dim_idx:dim_idx + 1]
    elif len(observation.shape) == 2:
      return observation[:, dim_idx:dim_idx + 1]
    elif len(observation.shape) == 3:
      return observation[:, :, dim_idx:dim_idx + 1]

  # for consistent use
  if FLAGS.reduced_observation == 0:
    return observation

  # process observation for dynamics with reduced observation space
  if FLAGS.environment == 'HalfCheetah-v1':
    qpos_dim = 9
  elif FLAGS.environment == 'Ant-v1':
    qpos_dim = 15
  elif FLAGS.environment == 'Humanoid-v1':
    qpos_dim = 26
  elif 'DKitty' in FLAGS.environment:
    qpos_dim = 36

  # x-axis
  if FLAGS.reduced_observation in [1, 5]:
    red_obs = [_shape_based_observation_processing(observation, 0)]
  # x-y plane
  elif FLAGS.reduced_observation in [2, 6]:
    if FLAGS.environment == 'Ant-v1' or 'DKitty' in FLAGS.environment or 'DClaw' in FLAGS.environment:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1)
      ]
    else:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, qpos_dim)
      ]
  # x-y plane, x-y velocities
  elif FLAGS.reduced_observation in [4, 8]:
    if FLAGS.reduced_observation == 4 and 'DKittyPush' in FLAGS.environment:
      # position of the agent + relative position of the box
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, 3),
          _shape_based_observation_processing(observation, 4)
      ]
    elif FLAGS.environment in ['Ant-v1']:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, qpos_dim),
          _shape_based_observation_processing(observation, qpos_dim + 1)
      ]

  # (x, y, orientation), works only for ant, point_mass
  elif FLAGS.reduced_observation == 3:
    if FLAGS.environment in ['Ant-v1', 'point_mass']:
      red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation,
                                              observation.shape[1] - 1)
      ]
    # x, y, z of the center of the block
    elif FLAGS.environment in ['HandBlock']:
      red_obs = [
          _shape_based_observation_processing(observation, 
                                              observation.shape[-1] - 7),
          _shape_based_observation_processing(observation, 
                                              observation.shape[-1] - 6),
          _shape_based_observation_processing(observation,
                                              observation.shape[-1] - 5)
      ]

  if FLAGS.reduced_observation in [5, 6, 8]:
    red_obs += [
        _shape_based_observation_processing(observation,
                                            observation.shape[1] - idx)
        for idx in range(1, 5)
    ]

  if FLAGS.reduced_observation == 36 and 'DKitty' in FLAGS.environment:
    red_obs = [
        _shape_based_observation_processing(observation, idx)
        for idx in range(qpos_dim)
    ]

  # x, y, z and the rotation quaternion
  if FLAGS.reduced_observation == 7 and FLAGS.environment == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(observation, observation.shape[-1] - idx)
        for idx in range(1, 8)
    ][::-1]

  # the rotation quaternion
  if FLAGS.reduced_observation == 4 and FLAGS.environment == 'HandBlock':
    red_obs = [
        _shape_based_observation_processing(observation, observation.shape[-1] - idx)
        for idx in range(1, 5)
    ][::-1]

  if isinstance(observation, np.ndarray):
    input_obs = np.concatenate(red_obs, axis=len(observation.shape) - 1)
  elif isinstance(observation, tf.Tensor):
    input_obs = tf.concat(red_obs, axis=len(observation.shape) - 1)
  return input_obs





def main(_):
  # setting up
  start_time = time.time()
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  global observation_omit_size, goal_coord, sample_count, iter_count, episode_size_buffer, episode_return_buffer

  root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)
  log_dir = os.path.join(root_dir, FLAGS.environment)
  
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)
  save_dir = os.path.join(log_dir, 'models')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)

  print('directory for recording experiment data:', log_dir)

  # in case training is paused and resumed, so can be restored
  try:
    sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
    iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
    episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
    episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
  except:
    sample_count = 0
    iter_count = 0
    episode_size_buffer = []
    episode_return_buffer = []

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(log_dir, 'train', 'in_graph_data'), flush_millis=10 * 1000)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(True):
    # environment related stuff
    unwrapped_env = get_environment(env_name=FLAGS.environment)
    py_env = wrap_env(
        skill_wrapper.SkillWrapper(
            unwrapped_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=None,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps)

    # all specifications required for all networks and agents
    py_action_spec = py_env.action_spec()
    tf_action_spec = tensor_spec.from_spec(
        py_action_spec)  # policy, critic action spec
    env_obs_spec = py_env.observation_spec()
    py_env_time_step_spec = ts.time_step_spec(
        env_obs_spec)  # replay buffer time_step spec
    if observation_omit_size > 0:
      agent_obs_spec = array_spec.BoundedArraySpec(
          (env_obs_spec.shape[0] - observation_omit_size,),
          env_obs_spec.dtype,
          minimum=env_obs_spec.minimum,
          maximum=env_obs_spec.maximum,
          name=env_obs_spec.name)  # policy, critic observation spec
    else:
      agent_obs_spec = env_obs_spec
    py_agent_time_step_spec = ts.time_step_spec(
        agent_obs_spec)  # policy, critic time_step spec
    tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

    if not FLAGS.reduced_observation:
      skill_dynamics_observation_size = (
          py_env_time_step_spec.observation.shape[0] - FLAGS.num_skills)
    else:
      skill_dynamics_observation_size = FLAGS.reduced_observation

    # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_agent_time_step_spec.observation,
        tf_action_spec,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
        (tf_agent_time_step_spec.observation, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      reweigh_batches_flag = True
    else:
      reweigh_batches_flag = False
    print()
    print_dict(time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. /
        (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)
    agent = dads_agent.DADSAgent(
        # DADS parameters
        save_dir,
        skill_dynamics_observation_size,
        observation_modify_fn=process_observation,
        restrict_input_size=observation_omit_size,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        normalize_observations=FLAGS.normalize_data,
        network_type=FLAGS.graph_type,
        num_mixture_components=FLAGS.num_components,
        fix_variance=FLAGS.fix_variance,
        reweigh_batches=reweigh_batches_flag,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
        # SAC parameters
        time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. /
        (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)

    # evaluation policy
    eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

    # collection policy
    if FLAGS.collect_policy == 'default':
      collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
    elif FLAGS.collect_policy == 'ou_noise':
      collect_policy = py_tf_policy.PyTFPolicy(
          ou_noise_policy.OUNoisePolicy(
              agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))

    # constructing a replay buffer, need a python spec
    policy_step_spec = policy_step.PolicyStep(
        action=py_action_spec, state=(), info=())

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      policy_step_spec = policy_step_spec._replace(
          info=policy_step.set_log_probability(
              policy_step_spec.info,
              array_spec.ArraySpec(
                  shape=(), dtype=np.float32, name='action_log_prob')))

    trajectory_spec = from_transition(py_env_time_step_spec, policy_step_spec,
                                      py_env_time_step_spec)
    capacity = FLAGS.replay_buffer_capacity
    # for all the data collected
    rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        capacity=capacity, data_spec=trajectory_spec)

    if FLAGS.train_skill_dynamics_on_policy:
      # for on-policy data (if something special is required)
      on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
          capacity=FLAGS.initial_collect_steps + FLAGS.collect_steps + 10,
          data_spec=trajectory_spec)

    # insert experience manually with relabelled rewards and skills
    agent.build_agent_graph()
    agent.build_skill_dynamics_graph()
    agent.create_savers()

    # saving this way requires the saver to be out the object
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'agent'),
        agent=agent,
        global_step=global_step)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'policy'),
        policy=agent.policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=rbuffer)

    skill_choice = np.random.uniform(low=-1, high=1, size=2)
    policy = agent.policy
    new_env = wrap_env(skill_wrapper.SkillWrapper(
            unwrapped_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=skill_choice,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
            max_episode_steps=FLAGS.max_env_steps)
    # py_env._test(skill_choice)
    obs = new_env.reset()
    with tf.compat.v1.Session().as_default() as sess:
      train_checkpointer.initialize_or_restore(sess)
      rb_checkpointer.initialize_or_restore(sess)
      agent.set_sessions(
          initialize_or_restore_skill_dynamics=True, session=sess)
      print(policy.action(hide_coords(obs)).action.eval())

class ExplorationAgent:

  def __init__(self, log_dir, skill=None):
    # setting up
    start_time = time.time()
    tf.compat.v1.enable_resource_variables()
    tf.compat.v1.disable_eager_execution()
    logging.set_verbosity(logging.WARNING)
    global observation_omit_size, goal_coord, sample_count, iter_count, episode_size_buffer, episode_return_buffer

    root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
    if not tf.io.gfile.exists(root_dir):
      tf.io.gfile.makedirs(root_dir)
    log_dir = os.path.join(root_dir, FLAGS.environment)
    
    if not tf.io.gfile.exists(log_dir):
      tf.io.gfile.makedirs(log_dir)
    save_dir = os.path.join(log_dir, 'models')
    if not tf.io.gfile.exists(save_dir):
      tf.io.gfile.makedirs(save_dir)

    print('directory for recording experiment data:', log_dir)

    # in case training is paused and resumed, so can be restored
    try:
      sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
      iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
      episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
      episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
    except:
      sample_count = 0
      iter_count = 0
      episode_size_buffer = []
      episode_return_buffer = []

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(log_dir, 'train', 'in_graph_data'), flush_millis=10 * 1000)
    train_summary_writer.set_as_default()

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(True):
      # environment related stuff
      unwrapped_env = get_environment(env_name=FLAGS.environment)
      py_env = wrap_env(
          skill_wrapper.SkillWrapper(
              unwrapped_env,
              num_latent_skills=FLAGS.num_skills,
              skill_type=FLAGS.skill_type,
              preset_skill=None,
              min_steps_before_resample=FLAGS.min_steps_before_resample,
              resample_prob=FLAGS.resample_prob),
          max_episode_steps=FLAGS.max_env_steps)

      # all specifications required for all networks and agents
      py_action_spec = py_env.action_spec()
      tf_action_spec = tensor_spec.from_spec(
          py_action_spec)  # policy, critic action spec
      env_obs_spec = py_env.observation_spec()
      py_env_time_step_spec = ts.time_step_spec(
          env_obs_spec)  # replay buffer time_step spec
      if observation_omit_size > 0:
        agent_obs_spec = array_spec.BoundedArraySpec(
            (env_obs_spec.shape[0] - observation_omit_size,),
            env_obs_spec.dtype,
            minimum=env_obs_spec.minimum,
            maximum=env_obs_spec.maximum,
            name=env_obs_spec.name)  # policy, critic observation spec
      else:
        agent_obs_spec = env_obs_spec
      py_agent_time_step_spec = ts.time_step_spec(
          agent_obs_spec)  # policy, critic time_step spec
      tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

      if not FLAGS.reduced_observation:
        skill_dynamics_observation_size = (
            py_env_time_step_spec.observation.shape[0] - FLAGS.num_skills)
      else:
        skill_dynamics_observation_size = FLAGS.reduced_observation

      # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
      actor_net = actor_distribution_network.ActorDistributionNetwork(
          tf_agent_time_step_spec.observation,
          tf_action_spec,
          fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
          continuous_projection_net=_normal_projection_net)

      critic_net = critic_network.CriticNetwork(
          (tf_agent_time_step_spec.observation, tf_action_spec),
          observation_fc_layer_params=None,
          action_fc_layer_params=None,
          joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

      if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
        reweigh_batches_flag = True
      else:
        reweigh_batches_flag = False
      print()
      # print_dict(time_step_spec=tf_agent_time_step_spec,
      #     action_spec=tf_action_spec,
      #     actor_network=actor_net,
      #     critic_network=critic_net,
      #     target_update_tau=0.005,
      #     target_update_period=1,
      #     actor_optimizer=tf.compat.v1.train.AdamOptimizer(
      #         learning_rate=FLAGS.agent_lr),
      #     critic_optimizer=tf.compat.v1.train.AdamOptimizer(
      #         learning_rate=FLAGS.agent_lr),
      #     alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
      #         learning_rate=FLAGS.agent_lr),
      #     td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      #     gamma=FLAGS.agent_gamma,
      #     reward_scale_factor=1. /
      #     (FLAGS.agent_entropy + 1e-12),
      #     gradient_clipping=None,
      #     debug_summaries=FLAGS.debug,
      #     train_step_counter=global_step)
      agent = dads_agent.DADSAgent(
          # DADS parameters
          save_dir,
          skill_dynamics_observation_size,
          observation_modify_fn=process_observation,
          restrict_input_size=observation_omit_size,
          latent_size=FLAGS.num_skills,
          latent_prior=FLAGS.skill_type,
          prior_samples=FLAGS.random_skills,
          fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
          normalize_observations=FLAGS.normalize_data,
          network_type=FLAGS.graph_type,
          num_mixture_components=FLAGS.num_components,
          fix_variance=FLAGS.fix_variance,
          reweigh_batches=reweigh_batches_flag,
          skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
          # SAC parameters
          time_step_spec=tf_agent_time_step_spec,
          action_spec=tf_action_spec,
          actor_network=actor_net,
          critic_network=critic_net,
          target_update_tau=0.005,
          target_update_period=1,
          actor_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=FLAGS.agent_lr),
          critic_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=FLAGS.agent_lr),
          alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
              learning_rate=FLAGS.agent_lr),
          td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
          gamma=FLAGS.agent_gamma,
          reward_scale_factor=1. /
          (FLAGS.agent_entropy + 1e-12),
          gradient_clipping=None,
          debug_summaries=FLAGS.debug,
          train_step_counter=global_step)
      self.agent = agent
      # evaluation policy
      eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

      # collection policy
      if FLAGS.collect_policy == 'default':
        collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
      elif FLAGS.collect_policy == 'ou_noise':
        collect_policy = py_tf_policy.PyTFPolicy(
            ou_noise_policy.OUNoisePolicy(
                agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))

      # constructing a replay buffer, need a python spec
      policy_step_spec = policy_step.PolicyStep(
          action=py_action_spec, state=(), info=())

      if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
        policy_step_spec = policy_step_spec._replace(
            info=policy_step.set_log_probability(
                policy_step_spec.info,
                array_spec.ArraySpec(
                    shape=(), dtype=np.float32, name='action_log_prob')))

      trajectory_spec = from_transition(py_env_time_step_spec, policy_step_spec,
                                        py_env_time_step_spec)
      capacity = FLAGS.replay_buffer_capacity
      # for all the data collected
      rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
          capacity=capacity, data_spec=trajectory_spec)

      if FLAGS.train_skill_dynamics_on_policy:
        # for on-policy data (if something special is required)
        on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
            capacity=FLAGS.initial_collect_steps + FLAGS.collect_steps + 10,
            data_spec=trajectory_spec)

      # insert experience manually with relabelled rewards and skills
      agent.build_agent_graph()
      agent.build_skill_dynamics_graph()
      agent.create_savers()

      # saving this way requires the saver to be out the object
      train_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(save_dir, 'agent'),
          agent=agent,
          global_step=global_step)
      self.policy_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(save_dir, 'policy'),
          policy=agent.policy,
          global_step=global_step)
      rb_checkpointer = common.Checkpointer(
          ckpt_dir=os.path.join(save_dir, 'replay_buffer'),
          max_to_keep=1,
          replay_buffer=rbuffer)

      
      
      # This is what we need
      self.policy = agent.policy
      if skill is None:
        self.set_rand_skill()
      else:
        self.set_skill(skill)

      with tf.compat.v1.Session().as_default() as sess:
        train_checkpointer.initialize_or_restore(sess)
        rb_checkpointer.initialize_or_restore(sess)
        agent.set_sessions(
            initialize_or_restore_skill_dynamics=True, session=sess)
    

  def compute_action(self, obs, **kwargs):
    with tf.compat.v1.Session().as_default() as sess:
      # self.agent.set_sessions(
      #     initialize_or_restore_skill_dynamics=True, session=sess)
      self.policy_checkpointer.initialize_or_restore(sess)
      new_obs = np.concatenate((obs, self.skill))
      new_obs = ts.transition(new_obs, 0)
      return self.policy.action(hide_coords(new_obs)).action.eval()

  def set_skill(self, skill):
    self.skill = skill

  def set_rand_skill(self):
    self.skill = np.random.uniform(low=-1, high=1, size=2)

  def get_policy(self):
    return self.policy

def main2(_):
  

  agent = ExplorationAgent('/Users/ericweiner/Documents/cfrl-rllib/dads_policies/DrivingPLE-v0')
  env = driving_creator()
  obs = env.reset()
  action = agent.compute_action(obs)
  print(action)


if __name__ == "__main__":
  # main2("IDIJFKSJDFLKJLJSDFLKSDKJIJFELDKJFLKJTENSOFLOW")
  tf.compat.v1.app.run(main2)


"""
  # setting up
  # start_time = time.time()
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()
  # logging.set_verbosity(logging.INFO)
  global observation_omit_size, goal_coord, sample_count, iter_count, episode_size_buffer, episode_return_buffer

  root_dir = os.path.abspath(os.path.expanduser(FLAGS.logdir))
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)
  log_dir = os.path.join(root_dir, FLAGS.environment)
  
  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)
  save_dir = os.path.join(log_dir, 'models')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)

  print('directory for recording experiment data:', log_dir)

  # in case training is paused and resumed, so can be restored
  try:
    sample_count = np.load(os.path.join(log_dir, 'sample_count.npy')).tolist()
    iter_count = np.load(os.path.join(log_dir, 'iter_count.npy')).tolist()
    episode_size_buffer = np.load(os.path.join(log_dir, 'episode_size_buffer.npy')).tolist()
    episode_return_buffer = np.load(os.path.join(log_dir, 'episode_return_buffer.npy')).tolist()
  except:
    sample_count = 0
    iter_count = 0
    episode_size_buffer = []
    episode_return_buffer = []

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      os.path.join(log_dir, 'train', 'in_graph_data'), flush_millis=10 * 1000)
  train_summary_writer.set_as_default()

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(True):
    # environment related stuff
    unwrapped_env = get_environment(env_name=FLAGS.environment)
    py_env = wrap_env(
        skill_wrapper.SkillWrapper(
            unwrapped_env,
            num_latent_skills=FLAGS.num_skills,
            skill_type=FLAGS.skill_type,
            preset_skill=None,
            min_steps_before_resample=FLAGS.min_steps_before_resample,
            resample_prob=FLAGS.resample_prob),
        max_episode_steps=FLAGS.max_env_steps)

    # all specifications required for all networks and agents
    py_action_spec = py_env.action_spec()
    tf_action_spec = tensor_spec.from_spec(
        py_action_spec)  # policy, critic action spec
    env_obs_spec = py_env.observation_spec()
    py_env_time_step_spec = ts.time_step_spec(
        env_obs_spec)  # replay buffer time_step spec
    if observation_omit_size > 0:
      agent_obs_spec = array_spec.BoundedArraySpec(
          (env_obs_spec.shape[0] - observation_omit_size,),
          env_obs_spec.dtype,
          minimum=env_obs_spec.minimum,
          maximum=env_obs_spec.maximum,
          name=env_obs_spec.name)  # policy, critic observation spec
    else:
      agent_obs_spec = env_obs_spec
    py_agent_time_step_spec = ts.time_step_spec(
        agent_obs_spec)  # policy, critic time_step spec
    tf_agent_time_step_spec = tensor_spec.from_spec(py_agent_time_step_spec)

    if not FLAGS.reduced_observation:
      skill_dynamics_observation_size = (
          py_env_time_step_spec.observation.shape[0] - FLAGS.num_skills)
    else:
      skill_dynamics_observation_size = FLAGS.reduced_observation

    # TODO(architsh): Shift co-ordinate hiding to actor_net and critic_net (good for futher image based processing as well)
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_agent_time_step_spec.observation,
        tf_action_spec,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        continuous_projection_net=_normal_projection_net)

    critic_net = critic_network.CriticNetwork(
        (tf_agent_time_step_spec.observation, tf_action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=(FLAGS.hidden_layer_size,) * 2)

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      reweigh_batches_flag = True
    else:
      reweigh_batches_flag = False
    print()
    # print_dict(time_step_spec=tf_agent_time_step_spec,
    #     action_spec=tf_action_spec,
    #     actor_network=actor_net,
    #     critic_network=critic_net,
    #     target_update_tau=0.005,
    #     target_update_period=1,
    #     actor_optimizer=tf.compat.v1.train.AdamOptimizer(
    #         learning_rate=FLAGS.agent_lr),
    #     critic_optimizer=tf.compat.v1.train.AdamOptimizer(
    #         learning_rate=FLAGS.agent_lr),
    #     alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
    #         learning_rate=FLAGS.agent_lr),
    #     td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    #     gamma=FLAGS.agent_gamma,
    #     reward_scale_factor=1. /
    #     (FLAGS.agent_entropy + 1e-12),
    #     gradient_clipping=None,
    #     debug_summaries=FLAGS.debug,
    #     train_step_counter=global_step)
    agent = dads_agent.DADSAgent(
        # DADS parameters
        save_dir,
        skill_dynamics_observation_size,
        observation_modify_fn=process_observation,
        restrict_input_size=observation_omit_size,
        latent_size=FLAGS.num_skills,
        latent_prior=FLAGS.skill_type,
        prior_samples=FLAGS.random_skills,
        fc_layer_params=(FLAGS.hidden_layer_size,) * 2,
        normalize_observations=FLAGS.normalize_data,
        network_type=FLAGS.graph_type,
        num_mixture_components=FLAGS.num_components,
        fix_variance=FLAGS.fix_variance,
        reweigh_batches=reweigh_batches_flag,
        skill_dynamics_learning_rate=FLAGS.skill_dynamics_lr,
        # SAC parameters
        time_step_spec=tf_agent_time_step_spec,
        action_spec=tf_action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        target_update_tau=0.005,
        target_update_period=1,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=FLAGS.agent_lr),
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=FLAGS.agent_gamma,
        reward_scale_factor=1. /
        (FLAGS.agent_entropy + 1e-12),
        gradient_clipping=None,
        debug_summaries=FLAGS.debug,
        train_step_counter=global_step)

    # evaluation policy
    eval_policy = py_tf_policy.PyTFPolicy(agent.policy)

    # collection policy
    if FLAGS.collect_policy == 'default':
      collect_policy = py_tf_policy.PyTFPolicy(agent.collect_policy)
    elif FLAGS.collect_policy == 'ou_noise':
      collect_policy = py_tf_policy.PyTFPolicy(
          ou_noise_policy.OUNoisePolicy(
              agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))

    # constructing a replay buffer, need a python spec
    policy_step_spec = policy_step.PolicyStep(
        action=py_action_spec, state=(), info=())

    if FLAGS.skill_dynamics_relabel_type is not None and 'importance_sampling' in FLAGS.skill_dynamics_relabel_type and FLAGS.is_clip_eps > 1.0:
      policy_step_spec = policy_step_spec._replace(
          info=policy_step.set_log_probability(
              policy_step_spec.info,
              array_spec.ArraySpec(
                  shape=(), dtype=np.float32, name='action_log_prob')))

    trajectory_spec = from_transition(py_env_time_step_spec, policy_step_spec,
                                      py_env_time_step_spec)
    capacity = FLAGS.replay_buffer_capacity
    # for all the data collected
    rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        capacity=capacity, data_spec=trajectory_spec)

    if FLAGS.train_skill_dynamics_on_policy:
      # for on-policy data (if something special is required)
      on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
          capacity=FLAGS.initial_collect_steps + FLAGS.collect_steps + 10,
          data_spec=trajectory_spec)

    # insert experience manually with relabelled rewards and skills
    agent.build_agent_graph()
    agent.build_skill_dynamics_graph()
    agent.create_savers()

    # saving this way requires the saver to be out the object
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'agent'),
        agent=agent,
        global_step=global_step)
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'policy'),
        policy=agent.policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(save_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=rbuffer)
    
    # This is what we need
    self.policy = agent.policy
    if skill is None:
      self.set_rand_skill()
    else:
      self.set_skill(skill)

    with tf.compat.v1.Session().as_default() as sess:
      train_checkpointer.initialize_or_restore(sess)
      rb_checkpointer.initialize_or_restore(sess)
      agent.set_sessions(
          initialize_or_restore_skill_dynamics=True, session=sess)
"""