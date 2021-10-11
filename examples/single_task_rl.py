from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget, OpenBox
import numpy as np


class Agent(object):

    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        gripper = [np.random.randint(2)]  # Discrete open or close
        return np.concatenate([arm, gripper], axis=-1)


obs_config = ObservationConfig()
obs_config.set_all_low_dim(False)        # set low dim obs (joint pos, grip force etc.)
obs_config.set_all_high_dim(False)      # turn off all cameras (high dim obs)
obs_config.wrist_camera.rgb = True
obs_config.wrist_camera.depth = True
obs_config.set_custom_low_dim(True) 

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

training_steps = 1200
episode_length = 100

task = env.get_task(OpenBox, max_episode_length = episode_length, dense_rewards = True)
#task = env.get_task(ReachTarget, max_episode_length = episode_length)

agent = Agent(env.action_size)
step_count = 0

obs = None
terminate = True
for i in range(training_steps):
    if terminate:
        step_count = 0
        print('Reset Episode')
        descriptions, obs = task.reset()
        #print(descriptions)
    action = agent.act(obs)
    #print(action)
    obs, reward, terminate = task.step(action)
    step_count += 1
    #print(step_count)
print('Done')
env.shutdown()
