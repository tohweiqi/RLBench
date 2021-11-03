from typing import Union, Dict, Tuple

import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np


class RLBenchEnv(gym.Env):
    """An gym wrapper for RLBench."""
    """custom_low_dim: joint_positions, gripper_pose, task_low_dim_state + wrist_camera_matrix"""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None, 
                 action_mode: ActionMode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY), 
                 gripper_speed: float = 0.2, 
                 max_episode_length: int = 200,
                 dense_rewards: int = 0):
                 
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(False)
        obs_config.set_all_high_dim(False)
        
        
        if observation_mode == 'state':
            obs_config.set_custom_low_dim(True)
            
        elif observation_mode == 'vision':            
            obs_config.set_all_high_dim(True)
            obs_config.wrist_camera.point_cloud = False
            obs_config.front_camera.point_cloud = False
            obs_config.overhead_camera.point_cloud = False
            obs_config.left_shoulder_camera.point_cloud = False
            obs_config.right_shoulder_camera.point_cloud = False            
        elif observation_mode == 'vision_wrist':
            obs_config.wrist_camera.set_all(True)
            obs_config.wrist_camera.point_cloud = False
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_front':
            obs_config.front_camera.set_all(True)
            obs_config.front_camera.point_cloud = False
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_overhead':
            obs_config.overhead_camera.set_all(True)
            obs_config.overhead_camera.point_cloud = False
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_left_shoulder':
            obs_config.left_shoulder_camera.set_all(True)
            obs_config.left_shoulder_camera.point_cloud = False
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_right_shoulder':
            obs_config.right_shoulder_camera.set_all(True)
            obs_config.right_shoulder_camera.point_cloud = False
            obs_config.set_custom_low_dim(True)
            
        elif observation_mode == 'vision_rgb':
            obs_config.wrist_camera.rgb = True
            obs_config.front_camera.rgb = True
            obs_config.overhead_camera.rgb = True
            obs_config.left_shoulder_camera.rgb = True
            obs_config.right_shoulder_camera.rgb = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_wrist_rgb':
            obs_config.wrist_camera.rgb = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_front_rgb':
            obs_config.front_camera.rgb = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_overhead_rgb':
            obs_config.overhead_camera.rgb = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_left_shoulder_rgb':
            obs_config.left_shoulder_camera.rgb = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_right_shoulder_rgb':
            obs_config.right_shoulder_camera.rgb = True
            obs_config.set_custom_low_dim(True)
            
        elif observation_mode == 'vision_rgbd':
            obs_config.wrist_camera.rgb = True
            obs_config.front_camera.rgb = True
            obs_config.overhead_camera.rgb = True
            obs_config.left_shoulder_camera.rgb = True
            obs_config.right_shoulder_camera.rgb = True
            obs_config.wrist_camera.depth = True
            obs_config.front_camera.depth = True
            obs_config.overhead_camera.depth = True
            obs_config.left_shoulder_camera.depth = True
            obs_config.right_shoulder_camera.depth = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_wrist_rgbd':
            obs_config.wrist_camera.rgb = True
            obs_config.wrist_camera.depth = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_front_rgbd':
            obs_config.front_camera.rgb = True
            obs_config.front_camera.depth = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_overhead_rgbd':
            obs_config.overhead_camera.rgb = True
            obs_config.overhead_camera.depth = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_left_shoulder_rgbd':
            obs_config.left_shoulder_camera.rgb = True
            obs_config.left_shoulder_camera.depth = True
            obs_config.set_custom_low_dim(True)
        elif observation_mode == 'vision_right_shoulder_rgbd':
            obs_config.right_shoulder_camera.rgb = True
            obs_config.right_shoulder_camera.depth = True
            obs_config.set_custom_low_dim(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
                
        #action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class, max_episode_length = max_episode_length, dense_rewards = dense_rewards)

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                "left_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_depth.shape),
                "right_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_depth.shape),
                "overhead_depth": spaces.Box(
                    low=0, high=1, shape=obs.overhead_depth.shape),
                "wrist_depth": spaces.Box(
                    low=0, high=1, shape=obs.wrist_depth.shape),
                "front_depth": spaces.Box(
                    low=0, high=1, shape=obs.front_depth.shape),
                "left_shoulder_mask": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_mask.shape),
                "right_shoulder_mask": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_mask.shape),
                "overhead_mask": spaces.Box(
                    low=0, high=1, shape=obs.overhead_mask.shape),
                "wrist_mask": spaces.Box(
                    low=0, high=1, shape=obs.wrist_mask.shape),
                "front_mask": spaces.Box(
                    low=0, high=1, shape=obs.front_mask.shape),
                })
        elif observation_mode == 'vision_wrist':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "wrist_depth": spaces.Box(
                    low=0, high=1, shape=obs.wrist_depth.shape),
                "wrist_mask": spaces.Box(
                    low=0, high=1, shape=obs.wrist_mask.shape),
                })
        elif observation_mode == 'vision_front':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                "front_depth": spaces.Box(
                    low=0, high=1, shape=obs.front_depth.shape),
                "front_mask": spaces.Box(
                    low=0, high=1, shape=obs.front_mask.shape),
                })
        elif observation_mode == 'vision_overhead':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                "overhead_depth": spaces.Box(
                    low=0, high=1, shape=obs.overhead_depth.shape),
                "overhead_mask": spaces.Box(
                    low=0, high=1, shape=obs.overhead_mask.shape),
                })
        elif observation_mode == 'vision_left_shoulder':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "left_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_depth.shape),
                "left_shoulder_mask": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_mask.shape),
                })
        elif observation_mode == 'vision_right_shoulder':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "right_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_depth.shape),
                "right_shoulder_mask": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_mask.shape),
                })
                
        elif observation_mode == 'vision_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                "left_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_depth.shape),
                "right_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_depth.shape),
                "overhead_depth": spaces.Box(
                    low=0, high=1, shape=obs.overhead_depth.shape),
                "wrist_depth": spaces.Box(
                    low=0, high=1, shape=obs.wrist_depth.shape),
                "front_depth": spaces.Box(
                    low=0, high=1, shape=obs.front_depth.shape),
                })
        elif observation_mode == 'vision_wrist_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "wrist_depth": spaces.Box(
                    low=0, high=1, shape=obs.wrist_depth.shape),
                })
        elif observation_mode == 'vision_front_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                "front_depth": spaces.Box(
                    low=0, high=0, shape=obs.front_depth.shape),
                })
        elif observation_mode == 'vision_overhead_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                "overhead_depth": spaces.Box(
                    low=0, high=1, shape=obs.overhead_depth.shape),
                })
        elif observation_mode == 'vision_left_shoulder_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "left_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_depth.shape),
                })
        elif observation_mode == 'vision_right_shoulder_rgbd':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "right_shoulder_depth": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_depth.shape),
                })
                        
        elif observation_mode == 'vision_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                })
        elif observation_mode == 'vision_wrist_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                })
        elif observation_mode == 'vision_front_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                })
        elif observation_mode == 'vision_overhead_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "overhead_rgb": spaces.Box(
                    low=0, high=1, shape=obs.overhead_rgb.shape),
                })
        elif observation_mode == 'vision_left_shoulder_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                })
        elif observation_mode == 'vision_right_shoulder_rgb':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                })
        


        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'vision_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "overhead_rgb": obs.overhead_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
            }
        elif self._observation_mode == 'vision_wrist_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "wrist_rgb": obs.wrist_rgb,
            }
        elif self._observation_mode == 'vision_front_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
            }
        elif self._observation_mode == 'vision_overhead_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "overhead_rgb": obs.overhead_rgb,
            }
        elif self._observation_mode == 'vision_left_shoulder_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
            }
        elif self._observation_mode == 'vision_right_shoulder_rgb':
            return {
                "state": obs.get_low_dim_data(),
                "right_shoulder_rgb": obs.right_shoulder_rgb,
            }
        elif self._observation_mode == 'vision_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "overhead_rgb": obs.overhead_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
                "left_shoulder_depth": obs.left_shoulder_depth,
                "right_shoulder_depth": obs.right_shoulder_depth,
                "overhead_depth": obs.overhead_depth,
                "wrist_depth": obs.wrist_depth,
                "front_depth": obs.front_depth,
            }
        elif self._observation_mode == 'vision_wrist_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "wrist_rgb": obs.wrist_rgb,
                "wrist_depth": obs.wrist_depth,
            }
        elif self._observation_mode == 'vision_front_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
                "front_depth": obs.front_depth,
            }
        elif self._observation_mode == 'vision_overhead_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "overhead_rgb": obs.overhead_rgb,
                "overhead_depth": obs.overhead_depth,
            }
        elif self._observation_mode == 'vision_left_shoulder_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "left_shoulder_depth": obs.left_shoulder_depth,
            }
        elif self._observation_mode == 'vision_right_shoulder_rgbd':
            return {
                "state": obs.get_low_dim_data(),
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "right_shoulder_depth": obs.right_shoulder_depth,
            }
            
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "overhead_rgb": obs.overhead_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
                "left_shoulder_depth": obs.left_shoulder_depth,
                "right_shoulder_depth": obs.right_shoulder_depth,
                "overhead_depth": obs.overhead_depth,
                "wrist_depth": obs.wrist_depth,
                "front_depth": obs.front_depth,
                "left_shoulder_mask": obs.left_shoulder_mask,
                "right_shoulder_mask": obs.right_shoulder_mask,
                "overhead_mask": obs.overhead_mask,
                "wrist_mask": obs.wrist_mask,
                "front_mask": obs.front_mask,
            }
        elif self._observation_mode == 'vision_wrist':
            return {
                "state": obs.get_low_dim_data(),
                "wrist_rgb": obs.wrist_rgb,
                "wrist_depth": obs.wrist_depth,
                "wrist_mask": obs.wrist_mask,
            }
        elif self._observation_mode == 'vision_front':
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
                "front_depth": obs.front_depth,
                "front_mask": obs.front_mask,
            }
        elif self._observation_mode == 'vision_overhead':
            return {
                "state": obs.get_low_dim_data(),
                "overhead_rgb": obs.overhead_rgb,
                "overhead_depth": obs.overhead_depth,
                "overhead_mask": obs.overhead_mask,
            }
        elif self._observation_mode == 'vision_left_shoulder':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "left_shoulder_depth": obs.left_shoulder_depth,
                "left_shoulder_mask": obs.left_shoulder_mask,
            }
        elif self._observation_mode == 'vision_right_shoulder':
            return {
                "state": obs.get_low_dim_data(),
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "right_shoulder_depth": obs.right_shoulder_depth,
                "right_shoulder_mask": obs.right_shoulder_mask,
            }


    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        obs, reward, terminate = self.task.step(action)
        success = False
        if reward > 0:
            success = True
        return self._extract_obs(obs), reward, terminate, {'is_success': success}

    def close(self) -> None:
        self.env.shutdown()
