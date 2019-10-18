import gym
class MountainCar:
    def __init__(self):
        self.min_position=-1.2
        self.max_position=0.6
        self.max_speed=0.07
        self.goal_position=0.5
    #random mode
    def test(self,steps):
        env.reset()
        total_reward=0
        for s in range(steps):
            env.render()
            action=env.action_space.sample()#get action randomly
            _,reward,done,_=env.step(action)
            total_reward+=reward
            if done:
                break
        return total_reward
    #specific rule mode
    def step(self,steps):
        observation=env.reset()
        total_reward=0
        for s in range(steps):
            env.render()#render the env every step
            action=self.get_action(observation)#get action for car
            #0 push left, 1 no push, 2 push right
            observation,reward,done,_=env.step(action)
            total_reward+=reward
            if done:
                print("Episode finished after {} timesteps. Total reward:{}"
                        .format(s+1,total_reward))
                break
        return total_reward

    def get_action(self,observation):
        pos=observation[0] #position
        vel=observation[1] #velocity
        if pos<self.goal_position/4 and -self.max_speed/2<vel<0:
            return 0
        elif pos>self.goal_position-0.02 and 0<vel<self.max_speed/10:
            return 1
        elif vel>0:
            return 2
        else:
            return 1

if __name__=='__main__':
    env=gym.make('MountainCar-v0')
    steps=200
    random_reward=0
    total_reward=0
    for i in range(3):
        random_reward+=MountainCar().test(steps)
    print("Average reward of random mode equals to {}".format(random_reward/3))
    for i in range(5):
        total_reward+=MountainCar().step(steps)
    print("Average reward equals to {}".format(total_reward/5))
    env.close()
