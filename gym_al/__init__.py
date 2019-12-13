from gym.envs.registration import register

register(
    id='CartPoleAl-v2',
    entry_point='gym_al.envs.cartpole_env:CartPoleEnvAl',
    max_episode_steps=200,
    reward_threshold=195.0,
)
register(
    id='MountainCarAl-v2',
    entry_point='gym_al.envs.mountain_car:MountainCarEnv',
    max_episode_steps=1000,
    reward_threshold=-300,
)
register(
    id='PendulumAl-v2',
    entry_point='gym_al.envs.pendulum:PendulumEnv',
    max_episode_steps=200,
)