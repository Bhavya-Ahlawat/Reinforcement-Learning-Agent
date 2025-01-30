import gym
import numpy as np
import random
import time

env = gym.make('CartPole-v1')

# Define discretization parameters for each state variable
pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-4, 4, 10)
ang_space = np.linspace(-.2095, .2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)

q_table = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n))

learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay_rate = 0.001
num_episodes = 5000  # Increased for better learning

def discretize_state(state):
    pos, vel, ang, ang_vel = state
    pos_d = np.digitize(pos, pos_space)
    vel_d = np.digitize(vel, vel_space)
    ang_d = np.digitize(ang, ang_space)
    ang_vel_d = np.digitize(ang_vel, ang_vel_space)
    return (pos_d, vel_d, ang_d, ang_vel_d)

for i in range(num_episodes):
    state = env.reset()[0]
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        state_index = discretize_state(state)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_index])

        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        next_state_index = discretize_state(next_state)

        q_table[state_index][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state_index]) - q_table[state_index][action])

        state = next_state

    epsilon = max(epsilon - epsilon_decay_rate, 0.01)
    if i % 100 == 0:
        print(f"Episode: {i}, Total Reward: {total_reward}, Epsilon: {epsilon}")

env = gym.make('CartPole-v1', render_mode="human") #render_mode="human" is important
state = env.reset()[0]
total_reward = 0
done = False
truncated = False
while not done and not truncated:
    state_index = discretize_state(state)
    action = np.argmax(q_table[state_index])  # Choose the best action based on the learned Q-table

    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(0.02)  # Adjust delay as needed

print(f"Total reward for the trained agent: {total_reward}")

env.close()