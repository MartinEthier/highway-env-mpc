from gymnasium.envs.registration import register

register(
    id="highway-env-mpc-v0",
    entry_point="highway_env_mpc.envs:HighwayEnvMPC",
    order_enforce=False,
    disable_env_checker=True,
    apply_api_compatibility=False
)