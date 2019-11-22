import gym
import numpy as np
import matplotlib.pyplot as plt
env=gym.make('MountainCar-v0')
env.reset()
LEARNING_RATE=0.5
DISCOUNT=0.95
EPISODES=10000
SHOW_EVERY=200
#Q table
Q_TABLE_LEN=20
DISCRETE_OS_SIZE=[Q_TABLE_LEN]*len(env.observation_space.high)
discrete_os_win_size=(env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE
q_table=np.zeros(DISCRETE_OS_SIZE+[env.action_space.n])
#decay epsilon
epsilon=1
START_EPSILON_DECAYING=1
END_EPSILON_DECAYING=EPISODES//2
epsilon_decay_value=epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)
def get_discrete_state(state):
    discrete_state=(state-env.observation_space.low)//discrete_os_win_size
    return tuple(discrete_state.astype(int))

def take_epilon_greedy_action(state,epsilon):
    discrete_state=get_discrete_state(state)
    if np.random.random()<epsilon:
        action=np.random.randint(0,env.action_space.n)
    else:
        action=np.argmax(q_table[discrete_state])
    return action
#reward
ep_rewards=[]
aggr_ep_rewards={'ep':[],'avg':[],'min':[],'max':[]}
#train
for episode in range(EPISODES):
    ep_reward=0
    if episode%SHOW_EVERY==0:
        render=True
    else:
        render=False
    state=env.reset()
    done=False
    while not done:
        action=take_epilon_greedy_action(state,epsilon)
        next_state,reward,done,_=env.step(action)
        ep_reward+=reward
        if not done:
            td_target=reward+DISCOUNT*np.max(q_table[get_discrete_state(next_state)])
            q_table[get_discrete_state(state)][action]+=\
                LEARNING_RATE*(td_target-q_table[get_discrete_state(state)][action])
        elif next_state[0]>=0.5:
            q_table[get_discrete_state(state)][action]=0
        state=next_state

    if END_EPSILON_DECAYING>=episode>=START_EPSILON_DECAYING:
        epsilon-=epsilon_decay_value
    ep_rewards.append(ep_reward)
    if episode%SHOW_EVERY==0:
        print("Episode: {} Reward: {}".format(episode,ep_reward))
        avg_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.legend(loc='upper left')
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.show()
