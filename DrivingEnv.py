

import math
import pygame
import sys
from AgentCar import Car


class DrivingEnv:
    def __init__(self,show_visualisation = True,show_sensors=True,show_colliders = True):
        self.SCREEN_WIDTH = 1244
        self.SCREEN_HEIGHT = 1016
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH,self.SCREEN_HEIGHT))
        self.track = pygame.image.load("assets/track.png")
        self.car = Car(self.screen)
        self.car_group = pygame.sprite.GroupSingle(self.car)
        
        self.n_games = 0
        self.frame = 0
        self.total_reward = 0
        self.checkpoints_reached = -1
        self.last_checkpoint_time = 0
        self.max_steps = 1000
        self.finished = False
        self.lapped = False

        self.visualisation = show_visualisation
        self.show_sensors = show_sensors
        self.show_colliders = show_colliders
        self.action = [0,0]

        self.checkpoints = [
            (600,820),
            (700,820),
            (800,815),
            (900,810),
            (1020,700),
            (1015,530),
            (1000,400),
            (810,235),
            (600,215),
            (350,220),
            (250,400),
            (300,640),
            (305,800),
            (550,820) #Finish line.

        ]
    
    def reset(self):
        self.car = Car(self.screen)
        self.car_group = pygame.sprite.GroupSingle(self.car)

        self.frame = 0
        self.total_reward = 0
        self.action = [0,0]
        self.checkpoints_reached = -1
        self.last_checkpoint_time = 0
        self.finished = False
        self.lapped = False
        
    def calculate_reward(self):
        reward = 0

        reward += 0.01 #for staying alive.

        # Speed reward with higher weight and minimum threshold
        # Optimal speed range: 0.4-0.8 of max speed
        normalized_speed = self.car.speed/self.car.MAX_SPEED
        if normalized_speed < 0.2:
        # Too slow - linear penalty
            speed_reward = -2.0 * (0.2 - normalized_speed)  # Max penalty: -0.4
        elif normalized_speed <= 0.8:
        # Good speed range - quadratic reward peaks around 0.6
            optimal_speed = 0.6
            speed_reward = 2 * (1 - abs(normalized_speed - optimal_speed) / 0.4)
        else:
        # Too fast - small penalty for reckless driving
            speed_reward = -1.0 * (normalized_speed - 0.8)
        
        reward += speed_reward
        #Distance progress
        reward += max(0,(-self.curr_dist_to_checkpoint+self.prev_dist_to_checkpoint)*0.05)

        #for reaching checkpoints.
        if self.check_progress():
            # Scale checkpoint reward based on how long it took
            time_taken = self.frame - self.last_checkpoint_time
            time_bonus = max(5.0, 25.0 - time_taken * 0.1)  # Bonus for reaching quickly
            reward += 50.0 + time_bonus  # Base reward + time bonus
            if self.finished:
                self.finished = False
                self.lapped = True
                reward += 150
        
        # 6. PROGRESSIVE TIME PENALTY (starts earlier, grows slower)
        time_since_checkpoint = self.frame - self.last_checkpoint_time
        if time_since_checkpoint > 60:  # Start penalty earlier
            # Exponential penalty that grows more gradually
            time_penalty = 0.1 #min(3.0, 0.005 * (time_since_checkpoint - 60) ** 1.2)
            reward -= time_penalty
        
        # Sensor-based rewards (encourage staying in the middle of track)
        min_sensor_distance = min(self.car.sensor_distances)
        if min_sensor_distance > 0.5:
            reward += 0.2  # Small bonus for staying away from walls
        elif min_sensor_distance < 0.2:
            reward -= 0.5  # Penalty for getting too close to walls
           
        
        #large penalty for crashing.
        if not self.car.alive:
            reward -= 150
        
        return reward

    def check_progress(self):
        car_pos = self.car.rect.center

        

        for i,checkpoint in enumerate(self.checkpoints):
            if i>self.checkpoints_reached:
                distance = math.sqrt((car_pos[0] - checkpoint[0])**2 + (car_pos[1] - checkpoint[1])**2)
                if distance <=57:
                    self.checkpoints_reached = i
                    self.last_checkpoint_time = self.frame
                    if i==len(self.checkpoints)-1:
                        self.finished = True

                    return True
                return False   
        
        return False



    def play_next_step(self,action):
        self.frame +=1 
        if self.checkpoints_reached == len(self.checkpoints)-1:
            print("TRACK LAPPED!!")
            self.checkpoints_reached = -1

        
        cur_pos = self.car.rect.center
        next_checkpoint = self.checkpoints_reached + 1
        self.prev_dist_to_checkpoint = math.sqrt((cur_pos[0]-self.checkpoints[next_checkpoint][0])**2+(cur_pos[1]-self.checkpoints[next_checkpoint][1])**2)
        
        #Move based on action.
        self.screen.blit(self.track,(0,0))
        self.car.update(action,self.screen)
        
        cur_pos = self.car.rect.center
        self.curr_dist_to_checkpoint = math.sqrt((cur_pos[0]-self.checkpoints[next_checkpoint][0])**2+(cur_pos[1]-self.checkpoints[next_checkpoint][1])**2)
        #Get rewards.
        reward = self.calculate_reward()
        self.total_reward += reward

        done = False
        if not self.car.alive or self.frame>= self.max_steps:
            done = True
        
        #Visualise if needed.
        if self.visualisation:
            self.render()

        return reward , done , self.total_reward , {"checkpoint":self.checkpoints_reached,"finished":self.lapped}
    
    def draw_checkpoints(self):
        for i,point in enumerate(self.checkpoints):
            col = (0,255,255)
            if i == self.checkpoints_reached + 1:
                col = (140,238,140)
            pygame.draw.circle(self.screen,col,point,57)
    
    def render(self):

        self.screen.blit(self.track,(0,0))
        self.draw_checkpoints()
        self.car_group.draw(self.screen)
        if self.show_sensors:
            self.car.draw_sensors(self.screen)
        

        pygame.display.update()







