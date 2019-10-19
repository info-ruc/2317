import gym
from numpy import arcsin

def get_choice(observation) -> int:
    cart_pos = observation[0]
    if cart_pos < -2.3:
        return 1
    elif cart_pos > 2.3:
        return 0
    
    sin_angle = observation[2]
    arcsin_angle = arcsin(sin_angle) * 180 / 3.14
    if arcsin_angle < - 2:
        return 0
    elif arcsin_angle > 2:
        return 1
    
    pole_v = observation[3]
    if pole_v < 0.0:
        return 0
    else:
        return 1

if __name__ == '__main__':

    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        for step in range(100):
            env.render()
            action = get_choice(observation)
            observation, reward, done, info = env.step(action)
            if done:
                get_choice(observation)
                print("Episode finished after {} timesteps".format(step + 1), " ", i_episode, "reward: ", reward)
                break
    env.close()
    print("end")
