import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio

class qlearn():
    def __init__(self,vals):
        self.alpha = vals["alpha"]
        self.gamma = vals["gamma"]
        self.epsilon = vals['epsilon']
        self.decay_rate = vals['epsilon_decay_rate']
        self.min_epsilon = vals['min_epsilon']
        self.num_episodes = vals['num_episodes']
        self.max_steps = vals['max_steps']
        self.env = gym.make('Acrobot-v1')
        #self.env = gym.make('Acrobot-v1',render_mode='human')
        self.num_bins_per_dim = [10, 10, 10, 10, 10, 10]
    #get_bin creates, and return the segmentation of the space that the agent interacts with
    def get_bin(self):
        obs_high = self.env.observation_space.high
        obs_low = self.env.observation_space.low
        bins = [np.linspace(obs_low[i], obs_high[i], self.num_bins_per_dim[i]) for i in range(len(obs_high))]
        return bins
    #get_qt returns the current q_table
    def get_qt(self):
        q_table_shape = tuple(self.num_bins_per_dim + [self.env.action_space.n])
        q_table = np.random.uniform(low=-1, high=1, size=q_table_shape)
        return q_table
    #dis_state finds the current state in the q_table
    def dis_state(self,state,bins):
        discrete_state = []
        for i, observation in enumerate(state):
            discrete_observation = np.digitize(observation, bins[i]) - 1
            discrete_observation = max(0, min(discrete_observation, len(bins[i]) - 2))
            discrete_state.append(discrete_observation)
        return tuple(discrete_state)
    #Eqisode_run runs the eqisode, and trains the agent / updates the q_table
    def eqisode_run(self):
        bins = self.get_bin()
        q_table = self.get_qt()
        rewards_per_episode = []
        steps_per_episode = []
        success_rates = [] 
        for episode in range(self.num_episodes):
            state, info = self.env.reset()
            discreteS = self.dis_state(state,bins)
            terminated = False
            truncated = False
            episode_reward = 0
            episode_steps = 0
            while not terminated and not truncated and episode_steps < self.max_steps:
                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(q_table[discreteS])
                next_state, reward, terminated, truncated, info = self.env.step(action)
                discrete_next_state = self.dis_state(next_state, bins)
                if not terminated and not truncated and (episode_steps + 1) == self.max_steps:
                    reward = -500 
                    truncated = True
                current_q_value = q_table[discreteS + (action,)]
                if terminated:
                    target_q_value = reward
                else:
                    max_future_q = np.max(q_table[discrete_next_state])
                    target_q_value = reward + self.gamma * max_future_q
            
                q_table[discreteS + (action,)] += self.alpha * (target_q_value - current_q_value)
                discreteS = discrete_next_state
                episode_reward += reward
                episode_steps += 1
            rewards_per_episode.append(episode_reward)
            steps_per_episode.append(episode_steps)
        
            if episode_reward > -self.max_steps:
                success_rates.append(1)
            else:
                success_rates.append(0)
        
            self.epsilon = max(self.min_epsilon, self.epsilon * np.exp(-self.decay_rate * episode))

            if episode % 1000 == 0:
                avg_reward = np.mean(rewards_per_episode[-1000:])
                avg_steps = np.mean(steps_per_episode[-1000:])
                avg_success_rate = np.mean(success_rates[-1000:]) * 100
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Avg Success Rate: {avg_success_rate:.2f}%")
        return rewards_per_episode,steps_per_episode,q_table
    
    def policy_run(self,q_table,filename="acrobot_policy_performance.gif",fps=30):
        env = gym.make('Acrobot-v1',render_mode='rgb_array')
        bins = self.get_bin()
        frames = []

        state,info = env.reset()
        discrete_state = self.dis_state(state, bins)
        test_reward = 0
        test_steps = 0
        terminated = False
        truncated = False
        frames.append(env.render()) # Capture the initial frame

        while not terminated and not truncated and test_steps < self.max_steps:
            action = np.argmax(q_table[discrete_state]) # Always exploit the learned policy

            next_state, reward, terminated, truncated, info = env.step(action)
            discrete_state = self.dis_state(next_state, bins)
            test_reward += reward
            test_steps += 1
            
            frames.append(env.render()) # Capture frame after each step

            
            if not terminated and not truncated and test_steps == self.max_steps:
                 test_reward += (-500 - reward) 
                 truncated = True

        env.close()
        if frames:
            imageio.mimsave(filename, frames,fps=fps)
            print(f"Video saved to {filename} with total reward: {test_reward:.2f} and steps: {test_steps}")
        else:
            print("No frames captured for video.")
##############
def getvals():
    alpha = 0.1       
    gamma = 0.99      
    epsilon = 1.0     
    epsilon_decay_rate = 0.001 
    min_epsilon = 0.01 
    num_episodes = 10000
    max_steps = 500
    return {"alpha":alpha,"gamma":gamma,"epsilon":epsilon,"epsilon_decay_rate":epsilon_decay_rate,"min_epsilon":min_epsilon,"num_episodes":num_episodes,"max_steps":max_steps}
def plotter(vals,title,xl,yl):
    plt.figure(figsize=(12, 6))
    plt.plot(vals)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

if __name__ == "__main__":
    animate = True
    reward_p_e,steps_p_e,agent = qlearn(getvals()).eqisode_run()
    plotter(reward_p_e,"Rewards per Episode",'Episode','Reward')
    plotter(steps_p_e,"Steps per Episode",'Episode','Steps')
    while animate:
        ani = input("Would you like to create an example animation (Y/N):")
        if ani.upper() == "Y":
            qlearn(getvals()).policy_run(agent)
        elif ani.upper() == "N":
            print("All Done")
            animate = False
        else:
            print("Input not valid")