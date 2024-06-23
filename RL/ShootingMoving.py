from vizdoom import *
import random
import time
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np
from vizdoom import DoomGame
from matplotlib import pyplot as pp
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
game = DoomGame()
game.load_config('/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/defend_the_center.cfg')
game.init()
game.close()

class VizDoomGym(gym.Env):
    def __init__(self, render=False):
        super(VizDoomGym, self).__init__()
        self.game = DoomGame()
        self.game.load_config('/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/defend_the_center.cfg')

        # Rendering settings
        self.game.set_window_visible(render)
        self.game.init()

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        self.seed()

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state1 = self.game.get_state().screen_buffer
            img = self.grayscale(state1)
            info = {'ammo': self.game.get_state().game_variables[0]}
        else:
            img = np.zeros(self.observation_space.shape)
            info = {'ammo': 0}

        done = self.game.is_episode_finished()
        return img, reward, done, False, info

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.game.new_episode()
        state1 = self.game.get_state().screen_buffer
        observation = self.grayscale(state1)
        info = {}
        return observation, info

    def render(self, mode='human'):
        pass

    def grayscale(self, obs):
        gray = cv2.cvtColor(np.moveaxis(obs, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state1 = np.reshape(resize, (100, 160, 1))
        return state1

    def close(self):
        self.game.close()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_training_start(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True

CHECKPOINT_DIR = './train/trained_models_moving'
LOG_DIR = './logs/models_log_moving'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym()
env_checker.check_env(env)

# n_steps reps batch_size
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=1024)

# Training
# model.learn(total_timesteps=100000, callback=callback)

# TEST
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
model = PPO.load('./train/trained_models_moving/best_model_100000.zip')

env = VizDoomGym(render=True)
env = Monitor(env)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward over {10} episodes: {mean_reward:.2f}")

# so we can view visually
for episode in range(5):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action,_ = model.predict(obs)
        obs, reward, done, _, info= env.step(action)
        time.sleep(0.05)
        total_reward += reward
        time.sleep(0.02)

    print("total reward for epi {} is {} ".format(episode,total_reward))
    time.sleep(2)

