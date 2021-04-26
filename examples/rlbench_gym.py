import gym
import rlbench.gym

env = gym.make('reach_target-vision-v0', render_mode='rgb_array')

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    action = env.action_space.sample()
    print(action)
    obs, reward, terminate, _ = env.step(action)
    env.render(mode='rgb_array')  # Note: rendering increases step time.

print('Done')
env.close()
