import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import losses,optimizers,metrics
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from tqdm import tqdm
env=gym.make("MountainCar-v0")
env.reset()
#Settings
ACTION_SPACE_SIZE=env.action_space.n
REPLAY_MEMORY_SIZE=50000
MIN_REPLAY_MEMORY_SIZE=1000
MINIBATCH_SIZE=32
UPDATE_TARGET_EVERY=5
DISCOUNT=0.99
EPISODES=1000
epsilon=1
EPSILON_DECAY=0.995
MIN_EPSILON=0.001
ep_rewards=[]
AGGREGATE_STATS_EVERY=50
MIN_EPSILON=0.001

def create_model():
    model=models.Sequential()
    model.add(Dense(16,input_shape=(env.observation_space.shape)))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(ACTION_SPACE_SIZE))
    model.add(Activation('linear'))
    print(model.summary())
    model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['accuracy'])
    return model

class DQNAgent:
    def __init__(self):
        self.replay_memory=deque(maxlen=REPLAY_MEMORY_SIZE)
        self.model_prediction=create_model()
        self.model_target=create_model()
        self.model_target.set_weights(self.model_prediction.get_weights())
        self.target_update_counter=0    

    def update_replay_memory(self,transition):
        self.replay_memory.append(transition)

    def get_qs(self,state):
        return self.model_prediction.predict(np.array(state).reshape(-1,*state.shape))[0]

    def train(self,terminal_state,step):    
        if len(self.replay_memory)<MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch=random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states=np.array([transition[0] for transition in minibatch])
        current_qs_list=self.model_prediction.predict(current_states)   
        next_states=np.array([transition[3] for transition in minibatch])
        target_qs_list=self.model_target.predict(next_states)
    
        X=[]
        Y=[]
        for index,(current_state,action,reward,next_state,done) in enumerate(minibatch):
            if not done:
                max_target_q=np.max(target_qs_list[index])
                new_q=reward+DISCOUNT*max_target_q
            else:
                new_q=reward
            current_qs=current_qs_list[index]
            current_qs[action]=new_q
            X.append(current_state)
            Y.append(current_qs)

        self.model_prediction.fit(np.array(X),np.array(Y),batch_size=MINIBATCH_SIZE,
                                verbose=0,shuffle=False if terminal_state else None)
        if terminal_state:
            self.target_update_counter+=1 
        if self.target_update_counter>UPDATE_TARGET_EVERY:
            self.model_target.set_weights(self.model_prediction.get_weights())
            self.target_update_counter=0

agent=DQNAgent()
aggr_ep_rewards={'ep':[],'avg':[],'min':[],'max':[]}
for episode in tqdm(range(1,EPISODES+1),ascii=True,unit='episodes'):
    ep_reward=0
    step=1
    state=env.reset()
    done=False
    while not done:
        if np.random.random()>epsilon:
            action=np.argmax(agent.get_qs(state))
        else:
            action=np.random.randint(0,ACTION_SPACE_SIZE)
        next_state,reward,done,_=env.step(action)
        ep_reward+=reward
        agent.update_replay_memory((state,action,reward,next_state,done))
        agent.train(done,step)
        state=next_state
        step+=1

    ep_rewards.append(ep_reward)
    if epsilon>MIN_EPSILON:
        epsilon*=EPSILON_DECAY
        epsilon=max(MIN_EPSILON,epsilon)
    if not episode % AGGREGATE_STATS_EVERY or episode==1:
        average_reward=sum(ep_rewards[-AGGREGATE_STATS_EVERY:])\
                        /len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-AGGREGATE_STATS_EVERY:]))

env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.legend(loc='upper left')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.show()
agent.model_prediction.save('dqn.model')
