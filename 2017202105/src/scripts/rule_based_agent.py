import time

import gym
from numpy.ma import arcsin

Cart_Position_MIN = -2.4
Cart_Position_MAX = 2.4


def get_choice(observation) -> int:
    car_x = observation[0]
    car_v = observation[1]
    pole_alpha_sin = observation[2]
    pole_top_v = observation[3]
    arcsin_alpha = arcsin(pole_alpha_sin) * 180 / 3.14
    # log_info(car_x, car_v, pole_alpha_sin, pole_top_v, arcsin_alpha)
    if car_x - Cart_Position_MIN < 0.1:
        return 1
    elif Cart_Position_MAX - car_x < 0.1:
        return 0
    if arcsin_alpha < - 2:
        return 0
    elif arcsin_alpha > 2:
        return 1
    if pole_top_v < 0.0:
        return 0
    else:
        return 1


def log_info(car_x, car_v, pole_alpha_sin, pole_top_v, arcsin_alpha):
    print("小车位置: ", car_x)
    print("小车速度: ", car_v)
    print("木棍角度正弦值: ", pole_alpha_sin)
    print("木棍角度: ", arcsin_alpha)
    print("木棍角度变化率: ", pole_top_v)
    print()


if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation1 = env.reset()
        for step in range(1000):
            env.render()
            action = get_choice(observation1)
            # print(action)
            # print()
            observation1, reward, done, info = env.step(action)
            if done:
                get_choice(observation1)
                print("Episode finished after {} timesteps".format(step + 1), " ", i_episode, "reward: ", reward)
                time.sleep(1)
                break
    env.close()
    print("end")
