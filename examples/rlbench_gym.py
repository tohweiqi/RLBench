import gym
import rlbench.gym

training_steps = 120
episode_length = 40

env = gym.make('open_box-vision_front-v0', render_mode='human', max_episode_length = episode_length)

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
