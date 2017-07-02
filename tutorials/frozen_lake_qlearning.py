import gym 
import numpy as np
import time           

API_KEY = "sk_rQdUjIzcR1eUcA7YwgrVSw"

problem = 'FrozenLake-v0'
algo_name = 'sarsa'
env = gym.make(problem)
env = gym.wrappers.Monitor(env, algo_name, force=True)

np.random.seed(1)
Q = np.ones([env.observation_space.n, env.action_space.n])

n_episodes = 5000
n_iter = 2000
alpha = 0.8
gamma = 0.9
j = 0
score = []

start = time.time()
## Train
for j in range(n_episodes):

    d = False
    
    s = env.reset()

    t = 0
    
    for t in range(n_iter):
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(j+1)))
        s1, r, d, _ = env.step(a)
        
        if s is not None:
            if d:
                Q[s,a] = Q[s,a] + alpha*(r-Q[s,a])
            else:
                Q[s,a] = Q[s,a] + alpha*(r+gamma*np.max(Q[s1,:])-Q[s,a])             
            s = s1

        if d:
            if len(score) < 100:
                score.append(r)
            else:
                score[j % 100] = r
            print("Episode {} finished after {} timesteps with r={}. Running score: {}".format(j, r, t, np.mean(score)))
            break
        
env.close()    
gym.upload(algo_name, api_key=API_KEY)


stop = time.time()
print("Training completed... Showtime")
print("It took: {0} episodes and {1} minutes".format(j,(stop-start)/60))