import gymnasium as gym
import pprint, copy, math
import numpy as np
import casadi as csd
from matplotlib import pyplot as plt
import pygame
import cv2

env = gym.make("highway-env-mpc-v0", render_mode='rgb_array')
pprint.pprint(env.config)

def filter_obs(obs):
    # Remove irrelevant agents
    relevant_obs = obs[obs[:, 0] != 0]
    ego_obs = relevant_obs[0]
    other_obs = relevant_obs[1:]
    return ego_obs, other_obs

obs, _ = env.reset()
ego_obs, _ = filter_obs(obs)
N = 30 # horizon length
ego_actions = np.zeros((N, 2))
dt = 1.0 / env.config['policy_frequency'] # s
print(dt)

vehicle_length = 5.0 # m
vehicle_width = 2.0 # m
lane_width = 4.0 # m

for it in range(200):
    # Setup MPC optimization
    opti = csd.Opti()
    # Control variables
    acc = opti.variable(N) # Longitudinal acceleration
    delta = opti.variable(N) # Steering angle
    # State variables
    x = opti.variable(N+1)
    y = opti.variable(N+1)
    speed = opti.variable(N+1)
    heading = opti.variable(N+1)
    
    # Objective: Maximize x position at the end of the horizon
    c_acc = 0 # Doesn't make sense to put a penalty on acceleration since our goal is to go fast
    c_delta = 0.5
    c_jerk = 0
    opti.minimize(-x[N] + c_acc * csd.sum1(acc) + c_delta * csd.sum1(delta) + c_jerk * csd.sum1(csd.diff(acc)))
    
    # Kinematic bicycle model
    for k in range(N):
        beta = csd.arctan(0.5 * csd.tan(delta[k]))
        vx = speed[k] * csd.cos(heading[k] + beta)
        vy = speed[k] * csd.sin(heading[k] + beta)
        x_next = x[k] + vx * dt
        y_next = y[k] + vy * dt
        heading_next = heading[k] + speed[k] * csd.sin(beta) / (vehicle_length / 2) * dt
        speed_next = speed[k] + acc[k] * dt
        opti.subject_to(x[k+1] == x_next)
        opti.subject_to(y[k+1] == y_next)
        opti.subject_to(heading[k+1] == heading_next)
        opti.subject_to(speed[k+1] == speed_next)
    
    # Input constraints
    opti.subject_to(opti.bounded(-6, acc, 6))
    opti.subject_to(opti.bounded(-np.pi/3, delta, np.pi/3))
    
    # Speed constraint
    opti.subject_to(opti.bounded(0, speed, 40))
    
    # Heading constraint
    opti.subject_to(opti.bounded(-np.pi/2, heading, np.pi/2))
    
    # Off-road constraint
    opti.subject_to(opti.bounded(-lane_width/2, y, (env.config['lanes_count'] - 1/2) * lane_width))
    
    # Simulate future trajectory of all vehicles in the scene for collision constraint
    env_copy = copy.deepcopy(env)
    if env.viewer is not None:
        env.viewer.other_traj = []
    for k in range(N):
        obs_next, _, _, _, _ = env_copy.step(ego_actions[k])
        _, other_obs_next = filter_obs(obs_next)
        if env.viewer is not None:
            env.viewer.other_traj.append(other_obs_next[:, 1:3])

        other_x = other_obs_next[:, 1] + vehicle_length/2 * np.cos(other_obs_next[:, 5])
        other_y = other_obs_next[:, 2] + vehicle_length/2 * np.sin(other_obs_next[:, 5])

        # Compare to xy at k+1
        x_adj = x[k+1] + vehicle_length/2 * csd.cos(heading[k+1])
        y_adj = y[k+1] + vehicle_length/2 * csd.sin(heading[k+1])
        for i in range(other_x.shape[0]):
            opti.subject_to(csd.sqrt((x_adj - other_x[i])**2 + (y_adj - other_y[i])**2) >= vehicle_length)

    # Set state values for k=0
    opti.subject_to(x[0] == ego_obs[1])
    opti.subject_to(y[0] == ego_obs[2])
    opti.subject_to(speed[0] == math.sqrt(ego_obs[3]**2 + ego_obs[4]**2))
    opti.subject_to(heading[0] == ego_obs[5])

    # Warm start to previous solution
    opti.set_initial(acc, ego_actions[:, 0])
    opti.set_initial(delta, ego_actions[:, 1])
    
    # Solve the optimization
    opti.solver('ipopt')
    sol = opti.solve()
    
    if env.viewer is not None:
        env.viewer.ego_traj = np.stack((sol.value(x), sol.value(y)), axis=1)
    
    # Save full open-loop trajectory for behaviour prediction step
    ego_actions = np.stack((sol.value(acc), sol.value(delta)), axis=1)
    
    # Apply action at k=0 to the simulator
    action = ego_actions[0]
    print(action)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break

    # Remove unobserved rows
    ego_obs, _ = filter_obs(obs)
    
    render_img = env.render()
    # Save image
    cv2.imwrite(f"/home/martin/school/masters/ece_780/highway-env-mpc/render_images/img_{it:04}.png", render_img)
    
