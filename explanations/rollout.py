from ray.rllib.rollout import *
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls
from ray.rllib.env.base_env import _DUMMY_AGENT_ID

import copy


# All functions included are modifications of rllib's functions
def run(args, parser):
    config = {}
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no config given on command line!")

    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)

    # Set num_workers to be at least 2.
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])

    # Merge with `evaluation_config`.
    evaluation_config = copy.deepcopy(config.get("evaluation_config", {}))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings.
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    # print(config)
    # print(cls)
    # print(type(cls))
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)

    # Determine the video output directory.
    # Deprecated way: Use (--out|~/ray_results) + "/monitor" as dir.
    video_dir = None
    if args.monitor:
        video_dir = os.path.join(
            os.path.dirname(args.out or "")
            or os.path.expanduser("~/ray_results/"), "monitor")
    # New way: Allow user to specify a video output path.
    elif args.video_dir:
        video_dir = os.path.expanduser(args.video_dir)

    # Do the actual rollout.
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        rollout(agent, args.env, num_steps, num_episodes, saver,
                args.no_render, video_dir)
    return agent.get_policy(), config

class RolloutSaver:
    """Utility class for storing rollouts.
    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:
    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]
    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info, simulator_state=None):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info, simulator_state])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, simulator_state])
        self._total_steps += 1

def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=None,
            no_render=True,
            video_dir=None):
    policy_agent_mapping = default_policy_agent_mapping

    print("AGENT TYPE", type(agent))  # TODO: remove
    print("AGENT POLICY", agent.get_policy())  # TODO: remove
    if saver is None:
        saver = RolloutSaver()

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        print("IF CASE")
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        print("STUFF", policy_map.items())
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        print("ELSE CASE")
        env = gym.make(env_name)
        multiagent = False
        try:
            policy_map = {DEFAULT_POLICY_ID: agent.policy}
        except AttributeError:
            raise AttributeError(
                "Agent ({}) does not have a `policy` property! This is needed "
                "for performing (trained) agent rollouts.".format(agent))
        use_lstm = {DEFAULT_POLICY_ID: False}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym.wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                img = env.render(mode="rgb_array")
            saver.append_step(obs, action, img, reward, done, info, env.game_state.game.getGameStateSave())
            steps += 1
            obs = next_obs
        saver.end_rollout()
        print("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1


def rollout_env(agent,
            env,
            handoff_func,
            obs,
            saver=None,
            no_render=True,
            video_dir=None):
    policy_agent_mapping = default_policy_agent_mapping

    if saver is None:
        saver = RolloutSaver()

    try:
        policy_map = {DEFAULT_POLICY_ID: agent.policy}
    except AttributeError:
        raise AttributeError(
            "Agent ({}) does not have a `policy` property! This is needed "
            "for performing (trained) agent rollouts.".format(agent))

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym.wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda x: True,
            force=True)

    mapping_cache = {}  # in case policy_agent_mapping is stochastic
    saver.begin_rollout()
    prev_actions = DefaultMapping(
        lambda agent_id: action_init[mapping_cache[agent_id]])
    prev_rewards = collections.defaultdict(lambda: 0.)
    done = False
    reward_total = 0.0
    while not done:
        action_dict = {}
        a_obs = obs
        agent_id = _DUMMY_AGENT_ID
        if a_obs is not None:
            policy_id = mapping_cache.setdefault(
                agent_id, policy_agent_mapping(agent_id))
            a_action = agent.compute_action(
                a_obs,
                prev_action=prev_actions[agent_id],
                prev_reward=prev_rewards[agent_id],
                policy_id=policy_id)
            a_action = flatten_to_single_ndarray(a_action)
            action_dict[agent_id] = a_action
            prev_actions[agent_id] = a_action
            done = handoff_func(a_obs, a_action)
        action = action_dict

        action = action[_DUMMY_AGENT_ID]
        next_obs, reward, env_done, info = env.step(action)
        done = done or env_done
        prev_rewards[_DUMMY_AGENT_ID] = reward

        reward_total += reward
        if not no_render:
            img = env.render(mode="rgb_array")
        else:
            img = None
        saver.append_step(obs, action, img, reward, done, info, env.game_state.game.getGameStateSave())
        obs = next_obs
    saver.end_rollout()
    return env, obs, env_done