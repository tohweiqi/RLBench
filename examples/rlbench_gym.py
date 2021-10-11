import gym
import rlbench.gym
from PIL import Image
import numpy as np
import cv2

training_steps = 20
episode_length = 10

env = gym.make('open_box-vision_front_rgbd-v0', max_episode_length = episode_length, gripper_speed=1.0)

terminate = True
for i in range(training_steps):
    if terminate:
        print('Reset Episode')
        obs = env.reset()
    action = env.action_space.sample()
    #print(action)
    obs, reward, terminate, _ = env.step(action)
    #env.render(mode='human')  # Note: rendering increases step time.

print('Done')
env.close()

disp_img = obs['front_depth']*255
disp_img = disp_img.astype(np.uint8)
print(disp_img)
print(np.amin(disp_img), np.amax(disp_img))
image = Image.fromarray(disp_img)
image.save("image_test.png")
cv2.imshow('image', disp_img)
cv2.waitKey(0)


