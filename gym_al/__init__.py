from gym.envs.registration import register

register(
    id='CartPoleAl-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)