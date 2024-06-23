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


# game = DoomGame()
# game.load_config('/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy1.cfg')
# game.init()
# actions = np.identity(7, dtype=np.uint8)
# for episode in range(5):
#     game.new_episode()
#
#     while not game.is_episode_finished():
#         state = game.get_state()
#         img = state.screen_buffer
#         info = state.game_variables
#         reward = game.make_action(random.choice(actions), 4)
#         print("reward : ", reward)
#         time.sleep(0.02)
#
#     print("Result : ", game.get_total_reward())
#     time.sleep(2)
#
# game.close()
#
class VizDoomGym(gym.Env):
    def __init__(self, render=False, config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy1.cfg'):
        super(VizDoomGym, self).__init__()
        self.game = DoomGame()
        self.game.load_config(config)

        self.game.set_window_visible(render)
        self.game.init()

        self.observation_space = spaces.Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(7)

        self.damage_taken = 0
        self.hitcount = 0
        self.selected_weapon_ammo = 52

        self.seed()

    def step(self, action):
        actions = np.identity(7, dtype=np.uint8)

        # already existent reward function , closer u get higher u get rewarded
        movement_reward = self.game.make_action(actions[action], 4)

        tot_reward = 0

        if self.game.get_state():
            state1 = self.game.get_state().screen_buffer
            img = self.grayscale(state1)

            # reward shaping
            game_var = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_var

            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.selected_weapon_ammo
            self.selected_weapon_ammo = ammo

            tot_reward = movement_reward + damage_taken_delta*10 + hitcount_delta*200 + ammo_delta*5
            info = ammo

        else:
            img = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()
        return img, tot_reward, done, False, info

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.game.new_episode()

        self.damage_taken = 0
        self.hitcount = 0
        self.selected_weapon_ammo = 52

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

# Using curriculum learning

CHECKPOINT_DIR='./train/trained_models_travelling'
LOG_DIR = './logs/models_log_travelling'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = VizDoomGym(config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy1.cfg')
env_checker.check_env(env)

model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192, clip_range=.1, gamma=.95, gae_lambda=.9)
model.learn(total_timesteps=60000, callback=callback)

env = VizDoomGym(config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy2.cfg')
model.set_env(env)
model.learn(total_timesteps=40000, callback=callback)

env = VizDoomGym(config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy3.cfg')
model.set_env(env)
model.learn(total_timesteps=40000, callback=callback)

env = VizDoomGym(config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy4.cfg')
model.set_env(env)
model.learn(total_timesteps=40000, callback=callback)

env = VizDoomGym(config='/Users/akshayv/Desktop/RLShooting/ViZDoom/scenarios/deadly_corridor copy5.cfg')
model.set_env(env)
model.learn(total_timesteps=40000, callback=callback)


# TO TEST
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor
# model = PPO.load('./train/trained_models/best_model_10000.zip')
#
# env = VizDoomGym(render=True)
# env = Monitor(env)
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
#
# print(f"Mean reward over {10} episodes: {mean_reward:.2f}")
#
# for episode in range(5):
#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, _, info = env.step(action)
#         time.sleep(0.05)
#         total_reward += reward
#         time.sleep(0.02)
#
#     print("total reward for epi {} is {} ".format(episode, total_reward))
#     time.sleep(2)

