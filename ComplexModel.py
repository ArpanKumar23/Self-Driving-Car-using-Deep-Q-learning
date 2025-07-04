import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super().__init__() #This inherits a nn module , this super().init() initialises that.
        self.linear1 = nn.Linear(input_size,256)
        self.linear2 = nn.Linear(256,256)
        self.linear3 = nn.Linear(256,128)
        self.linear4 = nn.Linear(128,output_size)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def save(self,file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(),file_name)

class QTrainer:
    
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() #Loss function (Q_new - Q)^2


    def train_step(self,state_old,action,reward,next_state,done):
        
        #We want this function to be able to run with a list of tuples too
        
        state = torch.tensor(state_old, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        #If they had multiple values this makes them all into shape (n,x) as required by pytorch.
        
        if len(state.shape) == 1:
            
            #Only one state is passed. reshape to required (1,x) shape.
            
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predicted Q values with current state.

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][action[idx].item()] = Q_new  #We use target to calculate the MSE , Here action is not a one hot vector! Instead its the index of our preffered action so use it directly.
            
        
        #Q_new = R + gamma*max(next_predicted Q value) : only do this if not done.
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()
 