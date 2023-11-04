import torch
import random
import numpy as np
from snake_game_ai import SnakeGame, Direction, Point
from collections import deque
from model import Linear_QNet, Trainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.01

class Agent : 
    def __init__(self) : 
        self.n_games = 0
        self.eps = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game) : 
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        # Danger here means when we are on the edge of the board
        # Danger right case (check if the right cell according to my direction is out of the board):
        # which means when moving : 
        #   -right : you need to check point_d
        #   -left : you need to check point_u
        #   -up : you need to check point_r
        #   -down : you need to check point_l
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done) : 
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) : 
        if len(self.memory) > BATCH_SIZE : 
            sample = random.sample(self.memory, BATCH_SIZE)
        else :
            sample = self.memory
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) : 
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state) : 
        # we use eps greedy policy
        # The more games we have the smallest epsilon will get -> more exploitation
        self.eps = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.eps : 
            move = random.randint(0,2)
            final_move[move] = 1
        else : 
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train() : 
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    while True : 
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move) 
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done : 
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record : 
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record', record) 


if __name__ == '__main__' : 
    train()