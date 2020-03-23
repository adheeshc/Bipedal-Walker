import gym
from keras.models import load_model
import numpy as np

model = load_model('BipedalWalker_model2.h5')
env = gym.make('BipedalWalker-v2')

NUM_EPISODES = 1

for episode in range(NUM_EPISODES):
    curr_state = env.reset()
    done = False
    time_taken = 0
    while not done:
        env.render()
        time_taken +=1
        curr_state = np.reshape(curr_state,(1,-1))
        action = model.predict(curr_state)
        action=np.reshape(action,(4,1))
        next_state, reward, done, _ = env.step(action)
        state = next_state
    print('time taken:', time_taken)