import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

from fontTools.misc.timeTools import epoch_diff


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.

    Parameters:
        Q (numpy.ndarray): The Q-table, a 2D array where Q[s, a] represents the expected future reward
                           of taking action a in state s.
        s (int): The current state.
        a (int): The action taken in the current state.
        r (float): The reward received after taking action a in state s.
        sprime (int): The next state after taking action a in state s.
        alpha (float): The learning rate, a value between 0 and 1 that determines how much new information
                       overrides the old information.
        gamma (float): The discount factor, a value between 0 and 1 that determines the importance of future
                       rewards.

    Returns:
        numpy.ndarray: The updated Q-table.
    """
    Q[s,a] = Q[s,a] + alpha*(r + gamma*np.max(Q[sprime,:]) - Q[s,a])

    return Q

def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    probability = random.uniform(0,1)
    if probability < epsilone:
        return random.randint(0,3)
    else:
        return np.argmax(Q[s,:])



if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.1 # choose your own learning rate

    gamma = 0.9 # choose your own discount factor

    epsilon = 0.6 # choose your own exploration rate

    n_epochs = 15 # choose your own number of epochs
    max_itr_per_epoch = 100 # choose your own number of iterations per epoch
    rewards = []

    for e in range(n_epochs):
        r = 0

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            S = Sprime

            if done :
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards), "over", n_epochs, "epochs.")

    # plot the rewards in function of epochs

    epoch_number = np.array(len(rewards))
    for i in range(len(rewards)):
        epoch_number.put(i)

    plt.figure()
    plt.plot(epoch_number, rewards, color="k", label='Rewards in function of epochs')
    plt.grid(color='0.7', linestyle='-', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Rewards in function of epochs')
    plt.show()

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
