import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module) : 
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x) : 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name = 'model.pth') : 
        model_folder_path = './model'
        if not os.path.exists(model_folder_path) : 
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

        
class Trainer : 
    def __init__(self, model, lr, gamma) : 
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
    

    def train_step(self, state, action, reward, new_state, done) :
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1 : 
            state = state.unsqueeze(0)
            new_state = new_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = [done]
        # prediciton with current state
        # state : (n, 11)
        # pred : (n, 3)
        pred = self.model(state)
        # r + y * max(next_predicted Q value)
        target = pred.clone()
        for idx in range(len(done)) : 
            Q_new = reward[idx]
            if not done[idx] : 
                Q_new += self.gamma * torch.max(self.model(new_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
            
        self.optim.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optim.step()

