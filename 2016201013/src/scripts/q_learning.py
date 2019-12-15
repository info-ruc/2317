import pickle # 保存模型用
from collections import defaultdict
import gym  # 0.12.5
import numpy as np

# 默认将Action 0,1的价值初始化为0
Q = defaultdict(lambda: [0, 0])

env = gym.make('CartPole-v0')

def transform_state(state):
	"""将 position, velocity, angle, angular velovity 通过线性转换映射到 [0, 40] 范围内"""
	pos, v, angle, ang_v= state
	pos_low, v_low, angle_low, ang_v_low= env.observation_space.low
	pos_high, v_high,angle_high, ang_v_high = env.observation_space.high

	ad_pos= 20 * (pos-pos_low)/(pos_high)
	ad_v =  20* (v+3.4)/(3.4) 
	ad_angle = 20 * (angle-angle_low)/(angle_high) 
	ad_ang_v = 20*(ang_v+3.4)/3.4
	
	return int(ad_pos), int(ad_v),int(ad_angle),int(ad_ang_v)



lr, factor = 0.7, 0.95
episodes = 1000000  # 训练次数，可改，预计在500万次使得结果较为理想
score_list = []  # 记录所有分数
for i in range(episodes):
	s = transform_state(env.reset())
	score = 0
	while True:
		a = np.argmax(Q[s])
		# 训练刚开始，多一点随机性，以便有更多的状态
		if np.random.random() > i * 3 / episodes:
			a = np.random.choice([0, 1])
		# 执行动作
		next_s, reward, done, _ = env.step(a)
		next_s = transform_state(next_s)
		# 根据上面的公式更新Q-Table
		Q[s][a] = (1 - lr) * Q[s][a] + lr * (reward + factor * max(Q[next_s]))
		score += reward
		s = next_s
		if done:
			score_list.append(score)
			print('episode:', i, 'score:', score, 'max:', max(score_list))
			break
env.close()


with open('CartPole-v0-q-learning.pickle', 'wb') as f:
	pickle.dump(dict(Q), f)
	print('model saved')