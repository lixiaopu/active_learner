from gym.envs.registration import register

register(
    id='CartPoleAl-v2',
    entry_point='gym_al.envs.cartpole_env:CartPoleEnvAl',
    max_episode_steps=200,
    reward_threshold=195.0,
)
