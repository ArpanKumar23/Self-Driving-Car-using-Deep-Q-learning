import pygame 
import math



class Car(pygame.sprite.Sprite):
    def __init__(self,screen):
        super().__init__()
        self.original_image = pygame.image.load("assets/car2.png")
        self.image = pygame.transform.rotozoom(self.original_image,0,0.07)
        self.rect = self.image.get_rect(center=(490, 820))
        self.speed = 5
        self.MAX_SPEED = 8
        self.angle = 0
        self.rotation_vel = 3
        self.acceleration = 0.3
        self.deacceleration = 0.4
        self.friction = 0.2
        self.distance_travelled_last_step = 0
        self.alive = True
    
        self.original_width = self.original_image.get_width() * 0.07  # accounting for scale
        self.original_height = self.original_image.get_height() * 0.07

        self.sensor_length = 200
        self.sensor_angles = [-90,-75,-45,-20,0,20,45,75,90]
        self.sensor_distances = [1,1,1,1,1,1,1,1,1]
        self.sensor_endpoints = []
        

    def update(self,action,screen,show_colliders = True):
        #Action = [forward,direction] , forward - 1,0,-1 direction-1,0,-1  , accelerate , no acc , de-acc and turn left , straight , turn right
        cur_pos = self.rect.center
        if not self.alive:
            return 
        #Acceleration
        if action[0]==1:
            self.speed = min(self.speed+self.acceleration,self.MAX_SPEED)
        elif action[0]==-1:
            self.speed = max(self.speed-self.deacceleration,-4)
        else:
            #Apply friction
            if self.speed>0:
                self.speed = max(self.speed-self.friction,0)
            elif self.speed<0:
                self.speed = min(self.speed+self.friction,0)

         
        #Turning only when moving.
        
        if action[1]==1 and abs(self.speed)>0.1:
            self.angle -= self.rotation_vel
        if action[1]==-1 and abs(self.speed)>0.1:
            self.angle += self.rotation_vel
        
        #Movement
        self.angle = self.angle % 360
        rad_angle = math.radians(self.angle)
        
        self.rect.centerx += self.speed*math.cos(rad_angle)
        self.rect.centery += self.speed*math.sin(rad_angle)

        #Rotating the image and scaling it down.
        self.image = pygame.transform.rotozoom(self.original_image,-self.angle,0.07)
        self.rect = self.image.get_rect(center = self.rect.center)
        final_pos = self.rect.center

        #Calculate distance travelled.
        self.distance_travelled_last_step = math.sqrt((final_pos[0]-cur_pos[0])**2+(final_pos[1]-cur_pos[1])**2)

        #Check for collision.
        self.update_sensors(screen)
        self.collision(screen,show_colliders)

    def get_sensors(self,screen,draw_sensors=True):
        
        if draw_sensors:
            self.draw_sensors(screen)
        
        return self.sensor_distances
        
        
        
    def collision(self,screen,show_colliders):
        corners = self.get_rotated_corners()
        
        for point in corners:
            if screen.get_at(point) == pygame.Color(2, 105, 31, 255):
                self.alive = False
                print("CRASHED")
            if show_colliders:
                pygame.draw.circle(screen, (0, 255, 255, 0),point, 4)
        
        if not show_colliders:
            return

        #This happens only when we want to show colliders    
        if len(corners) == 4:
            for i in range(4):
                start_point = corners[i]
                end_point = corners[(i + 1) % 4]
                pygame.draw.line(screen, (255, 0, 0), start_point, end_point, 2)
    def get_rotated_corners(self):
        """Calculate the 4 corners of the rotated car"""
        # Get the center of the car
        center_x, center_y = self.rect.center
        
        # Half dimensions
        half_width = self.original_width / 2
        half_height = self.original_height / 2
        
        # Convert angle to radians
        rad_angle = math.radians(self.angle)
        cos_angle = math.cos(rad_angle)
        sin_angle = math.sin(rad_angle)
        
        # Calculate the 4 corners relative to center, then rotate them
        corners = []
        
        # Original corners relative to center (before rotation)
        relative_corners = [
            (-half_width, -half_height),  # top-left
            (half_width, -half_height),   # top-right
            (half_width, half_height),    # bottom-right
            (-half_width, half_height)    # bottom-left
        ]
        
        # Rotate each corner and translate to world position
        for rel_x, rel_y in relative_corners:
            # Apply rotation matrix
            rotated_x = rel_x * cos_angle - rel_y * sin_angle
            rotated_y = rel_x * sin_angle + rel_y * cos_angle
            
            # Translate to world position
            world_x = center_x + rotated_x
            world_y = center_y + rotated_y
            
            corners.append((int(world_x), int(world_y)))
        
        return corners
    
    def raycast(self,screen,start_x,start_y,angle,max_distance):
        rad_angle = math.radians(angle)
        dx , dy = math.cos(rad_angle),math.sin(rad_angle)

        step_size = 2
        distance = 0

        while distance<max_distance:

            ray_x = int(start_x + dx*distance)
            ray_y = int(start_y + dy*distance)

            pixel_color = screen.get_at((ray_x,ray_y))
            if pixel_color == pygame.Color(2, 105, 31, 255):
                break
        
            distance += step_size

        norm_distance = min(distance/max_distance,1.0)
        return norm_distance , (ray_x , ray_y)

    
    def update_sensors(self,screen):
        self.sensor_distances = []
        self.sensor_endpoints = []

        center_x , center_y = self.rect.center

        for sensor_angle in self.sensor_angles:
            absolute_angle = self.angle + sensor_angle

            distance , endpoint = self.raycast(screen,center_x,center_y,absolute_angle,self.sensor_length)

            self.sensor_distances.append(distance)
            self.sensor_endpoints.append(endpoint)
        
        
        
    
    def draw_sensors(self, screen):
        center_x, center_y = self.rect.center
        
        for i, endpoint in enumerate(self.sensor_endpoints):
            # Color based on distance (red = close obstacle, green = far/no obstacle)
            distance = self.sensor_distances[i]
            color_intensity = int(255 * (1 - distance))
            color = (color_intensity,255-color_intensity,0)
            
            # Draw the ray line
            pygame.draw.line(screen, color , (center_x, center_y), endpoint, 2)
            
            # Draw endpoint dot
            pygame.draw.circle(screen, color, endpoint, 3)
            
        
    

    
        



