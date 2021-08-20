from gym.envs.registration import register
register(
    id="optim-v0",
    entry_point="optim_env.envs:OptimEnv",
)