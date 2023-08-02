from typing import Dict, Text, List, Tuple, Optional

import numpy as np
import gymnasium as gym
import pygame

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle import kinematics, objects
from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics
from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics

Observation = np.ndarray


# class RoadWithPrediction(Road):
    
#     def __init__(self,
#                 network: RoadNetwork = None,
#                 vehicles: List['kinematics.Vehicle'] = None,
#                 road_objects: List['objects.RoadObject'] = None,
#                 np_random: np.random.RandomState = None,
#                 record_history: bool = False) -> None:
#         super().__init__(network, vehicles, road_objects, np_random, record_history)
    
#     def simulate_agents(self, ego_action, controlled_vehicle):
#         """
#         Simulate the vehicles in the scene
        
#         Create a copy of the road class and call 
#         """
#         print(self.vehicles)
#         ego = Vehicle.create_from(controlled_vehicle)
#         agents = [IDMVehicle.create_from(vehicle) for vehicle in self.vehicles if vehicle != controlled_vehicle]
#         #print(vehicles_copy)
        
#         return None

class NewViewer(EnvViewer):
    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None) -> None:
        super().__init__(env, config)
        self.ego_traj = None
        self.other_traj = None
    
    def display(self) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if EnvViewer.agent_display:
            EnvViewer.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)
        
        if self.ego_traj is not None:
            for xy in self.ego_traj:
                pygame.draw.circle(self.sim_surface, (255, 0, 0), [*self.sim_surface.pos2pix(xy[0], xy[1])], self.sim_surface.pix(0.5))
        if self.other_traj is not None:
            for traj in self.other_traj:
                for xy in traj:
                    pygame.draw.circle(self.sim_surface, (0, 0, 255), [*self.sim_surface.pos2pix(xy[0], xy[1])], self.sim_surface.pix(0.5))

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1

class HighwayEnvMPC(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                # "features_range": {
                #     "x": [-100, 100],
                #     "y": [-100, 100],
                #     "vx": [-20, 20],
                #     "vy": [-20, 20]
                # },
                "normalize": False,
                "absolute": True,
                "order": "sorted"
            },
            "action": {
                "type": "ContinuousAction",
            },
            "lanes_count": 3,
            "vehicles_count": 20,
            "controlled_vehicles": 1,
            "initial_lane_id": -1,
            "simulation_frequency": 10,
            "policy_frequency": 10,
            "duration": 60,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.6,
            "collision_reward": -1,     # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,   # The reward received when driving on the right-most lanes, linearly mapped to
                                        # zero for other lanes.
            "high_speed_reward": 0.4,   # The reward received when driving at full speed, linearly mapped to zero for
                                        # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,    # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False
        })
        return config


    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        #self.ego_traj = None

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation_type.observe()
        reward = self._reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.viewer is None:
            self.viewer = NewViewer(self)

        self.enable_auto_render = True

        self.viewer.display()
        # # self.ego_traj is (N, 2) x,y
        # if self.ego_traj is not None:
        #     print('------')
        #print(self.ego_traj)
        # ego_surface = pygame.Surface((self.viewer.sim_surface.pix(length), self.viewer.sim_surface.pix(length)),
        #                                  flags=pygame.SRCALPHA)  # per-pixel alpha
        

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
