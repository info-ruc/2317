import argparse
import sys
import gym
from gym import wrappers, logger

class random agent(object):
	def _init_(self,action_space):
		self.action_space=action_space
	def act(self,obervation,reward,done):
		return self.action_space.sample()

if __name_=='_main_':
	parse = argparse.ArgumentParser(descripition=None)
	parser.add_argument('env_id',nargs='?',default='CartPole-v0',help='Select the environment to run')
	args=parser.parser_args()
	logger.set_level(logger.INFG)
	env=gym.make(args.env_id)
	outdir='/tmp/random-agent-results'
	env=wrappers.Monitor(env,directory,video_callable=False,force=True)
	env.seed(0)
	agent=RandeomAgent(env.action_space)
	episode_count=100
	reward=0
	done=False
	
	for i in range(episode_count):
		ob=env.reset()
		while True:
			action=agent.act(ab,reward,done)
			ob,reward,done,_=env.step(action)				
			if done:
			break

	env.close()