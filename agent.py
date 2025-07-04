import torch
import random
import numpy as np
from collections import deque
from ComplexModel import Linear_QNet,QTrainer
from DrivingEnv import DrivingEnv
from helper import plot
import numpy as np
import copy
import math
import time

MAX_MEMORY = 100000
BATCH_SIZE = 1024
LR = 0.001
FRAME_SKIP = 4

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.9
        
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999                
        self.gamma = 0.99
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(12,256,7)
        self.trainer = QTrainer(self.model, lr=LR,gamma= self.gamma)
        self.frame_skip = FRAME_SKIP
        self.action_space = {
            0: [0,0],           # Do nothing (maintain current speed)
            1: [1,0],
            2:[-1,0],      # Speed up        
            3: [1,1],      # Accelerate + turn left (racing line)  
            4: [1,-1],
            5: [0,1],
            6:[0,-1]   
                 # Accelerate + turn right (racing line)
        }    
    def get_state(self,game):
        state = copy.deepcopy(game.car.sensor_distances)
        speed = game.car.speed/game.car.MAX_SPEED
        cos = math.cos(game.car.angle)
        sin = math.sin(game.car.angle)

        state.append(speed)
        state.append(cos)
        state.append(sin)

        return np.array(state , dtype=np.float32)
    
    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    
    def train_long_memory(self):

        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        #TODO train step.
        self.trainer.train_step(states,actions,rewards,next_states,dones)
    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    
    def get_action(self,state):
        """Epsilon greedy selection"""

        if random.random() < self.epsilon:
            action = random.randint(0,6)
        else:
            state_tensor = torch.tensor(state , dtype=torch.float)
            Q_values = self.model(state_tensor)
            action = torch.argmax(Q_values).item()
        
        return action 
    
    def update_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        


def Train():
    plot_scores = [] #Keeps track of score while training.
    plot_mean_scores = [] 
    total_score = 0
    record = 0
    completed_tracks = 0
    best_completion_score = 0

    agent = Agent()
    game = DrivingEnv(show_visualisation=True)

    start_time = time.time()
    while True:
        #Get old state
        state_old = agent.get_state(game)
        
        #Get action based on the state
        final_action = agent.get_action(state_old)
        
        total_reward = 0
        done = False
        
        for skip_frame in range(agent.frame_skip):
            if not done:  # Only continue if game isn't over
                reward, done, score, info = game.play_next_step(agent.action_space[final_action])
                total_reward += reward
                
                # Break early if episode ends during frame skip
                if done:
                    break
        
        # Get state after all skipped frames

        state_new = agent.get_state(game)
        #Train the short memory of the agent.
        agent.train_short_memory(state_old,final_action,total_reward,state_new,done)
        
        #Add to agents memory.
        agent.remember(state_old,final_action,total_reward,state_new,done)
        
        if done:
            end_time = time.time()
            game.n_games +=1 
            agent.n_games +=1 
            agent.update_epsilon()
            #TODO Learn about target networks and how to use nd why.
            agent.train_long_memory()

            track_completed = info["finished"]
            if track_completed:
                completed_tracks +=1
            



            #Visualise in intervals if needed.
            if agent.n_games >=30:
                game.visualisation = False
            if (agent.n_games+1)%10==0:
                game.visualisation = True

            if track_completed and score>best_completion_score:
                best_completion_score = score
                agent.model.save("Best_Finish_Model.pth")
            if score>record:
                record = score
                agent.model.save("best_Normal_model.pth")
            
            save_plot = 0
            if agent.n_games % 500 == 0:
                save_plot = agent.n_games
                model_name = "model_"+str(agent.n_games)+".pth"
                agent.model.save(model_name)
            
            print(f"Episode : {agent.n_games}, total_score = {score}, Checkpoints reached : {info["checkpoint"]},steps : {game.frame}, epsilon: {agent.epsilon}, record: {record},best finish:{best_completion_score}, finished: {completed_tracks} time_taken : {end_time-start_time} seconds")
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores,save_plot)

            game.reset()
            start_time = time.time()

if __name__=="__main__":
    Train()

