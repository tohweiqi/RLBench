import gym
import rlbench.gym

training_steps = 3000
episode_length = 300

env = gym.make('empty_container-vision_wrist-v0', render_mode='human', max_episode_length = episode_length, gripper_speed=1.0)

terminate = True
for i in range(training_steps):
    if terminate:
        print('Reset Episode')
        obs = env.reset()
    action = env.action_space.sample()
    #print(action)
    obs, reward, terminate, _ = env.step(action)
    env.render(mode='human')  # Note: rendering increases step time.

print('Done')
env.close()
