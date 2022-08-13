import numpy as np
import random
import math
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.animation as animations

random.seed(1)

class State:
    def __init__(self, curr_x:int, curr_y:int, goal_x:int, goal_y:int )->None:
        self.curr_x = curr_x
        self.curr_y = curr_y
        self.goal_x = goal_x
        self.goal_y = goal_y

    def __str__(self) -> str:
        return f"({self.curr_x}, {self.curr_y}, {self.goal_x}, {self.goal_y})"


class roadMap:
    ROW_COUNT = 10
    COL_COUNT = 10

    def __init__(self, person_init, goal)->None:
        self.person_init = person_init
        self.goal = goal
        # self.depots = dict()
    
    def next_cell(self, row, col, action):
        assert row>=0 and row<self.ROW_COUNT, "row out of range" 
        assert col>=0 and col<self.COL_COUNT, "col out of range"
        assert action in ['North', 'East', 'South', 'West'], "invalid action"

        #obstructions in the map, hardcoding the obstructions/walls
    
        if (row >=0 and row <=3) and (col == 0 or col ==3 or col ==7) and action == 'East':
            return (row, col)
        if (row >=6 and row <=9) and (col ==2 or col== 7) and action == 'East':
            return (row, col)
        if (row >=4 and col <= 7) and (col==5) and action == 'East':
            return (row, col)
        if (row >=0 and row <=3) and (col == 1 or col ==4 or col ==8) and action == 'West':
            return (row, col)
        if (row >=6 and row <=9) and (col ==3 or col== 8) and action == 'West':
            return (row, col)
        if (row >=4 and col <= 7) and (col==6) and action == 'West':
            return (row, col)
    
        if action == 'North':
            if (row + 1 >= self.ROW_COUNT):
                return (row, col)
            return (row+1, col)
        elif action == 'East':
            if (col + 1 >= self.COL_COUNT):
                return (row, col)
            return (row, col+1)
        elif action == 'South':
            if (row - 1 < 0):
                return (row, col)
            return (row-1, col)
        elif action == 'West':
            if (col - 1 < 0):
                return (row, col)
            return (row, col-1)



class MDPModel:
    def __init__(self, roadMap, alpha = 0.25, epsilon = 0.1, decaying = False, max_steps = 500, max_episodes = 5000, discount_fator = 0.99) -> None:
        self.roadMap = roadMap
        self.epsilon = epsilon
        self.alpha = alpha
        self.decaying = decaying
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.discount_fator = discount_fator
        self.actions = ['North', 'East', 'South', 'West']
        self.states = dict()
        self.states_idx = dict()
        self.Q_sa = dict()
        self.policy = dict()
        self.step_num = 1
        self.episode_num = 0
        self.episodes = range(1000,5000,100)
        self.discounter_rewards =  []
        self.init_states()
        


    def states_value_initialisation(self)->None:
        for row in range(self.roadMap.ROW_COUNT):
            for col in range(self.roadMap.COL_COUNT):
                for goal_row in range(self.roadMap.ROW_COUNT):
                    for goal_col in range(self.roadMap.COL_COUNT):
                        self.states[(row, col, goal_row, goal_col)] = State(row, col, goal_row, goal_col)
        for i,state in enumerate(self.states.values()):
            self.states_idx[state] = i
            for action in self.actions:
                self.Q_sa[(state,action)] = 0


    def get_distance(self, state:State)->float:
        return float(abs(abs(state.curr_x - state.goal_x)**2 + abs(state.curr_y - state.goal_y)**2)**0.5)


    def get_reward(self, state:State, action, next_state:State)->float:
        assert action in self.actions, "invalid action" 
        if next_state.curr_x == self.goal.curr_x and next_state.curr_y == self.goal.curr_y:
            return +20
        elif next_state.curr_x == state.curr_x and next_state.curr_y == state.curr_y:
            return -10
        elif self.get_distance(next_state) < self.get_distance(state):
            return +4
        else:
            return -2


    def get_next_state(self, state:State, action)->State:
        next_coord = self.roadMap.next_cell(state.curr_x, state.curr_y, action)
        if next_coord == (state.curr_x, state.curr_y):
            return state
        else:
            return self.states[(next_coord[0], next_coord[1], state.goal_x, state.goal_y)]


    def generate_random_state(self) -> State:
        return np.random.choice(list(self.states.values()))
    

    def get_action(self, action_suggested)->str:
        probs = [0.05, 0.05,0.05, 0.05]
        dir = ['North', 'East', 'South', 'West']
        probs[dir.index(action_suggested)] = 0.85
        return np.random.choice(dir, p=probs)


    def get_greedyAction(self, state:State) -> str:
        max_val = -math.inf
        max_acts = []
        for action in self.actions:
            max_val = max(max_val, self.Q_sa[(state, action)])
        for action in self.actions:
            if self.Q_sa[(state, action)] == max_val:
                max_acts.append(action)
        return np.random.choice(max_acts)
    

    def get_epsilonAction(self, state:State) -> str:
        action = self.get_greedyAction(state)
        eps = self.epsilon
        if self.decaying:
            eps = eps * (1 - self.episode_num / self.max_episodes)
        which = np.random.choice(['greedy','random'], p = [1-eps,eps])
        if (which == 'greedy'):
            return action
        else:
            probs = np.ones((len(self.actions),))
            probs/=len(self.actions)
            return np.random.choice(self.actions,p= probs)


    def run_episode(self):
        self.episode_num +=1
        curr_state = self.generate_random_state()
        curr_action = self.get_epsilonAction(curr_state)
        iter =0
        while (iter < self.max_steps):
            action_taken = self.get_action(curr_action)
            next_state =  self.get_next_state(curr_state, action_taken)
            reward = self.get_reward(curr_state, curr_action, next_state)
            if (self.sarsa == True):
                next_action = self.get_epsilonAction(next_state)
            else:
                next_action = self.get_greedyAction(next_state)
                
            # now updating the Q value
            if (curr_state == self.states[(self.roadMap.goal[0], self.roadMap.goal[1], self.roadMap.goal[0], self.roadMap.goal[1])]):
                self.Q_sa[(curr_state, curr_action)] = reward 
                break
            else:
                sample= reward + self.discount_factor*self.Q_sa[(next_state, next_action)]
                self.Q_sa[(curr_state, curr_action)] = (1-self.alpha)*self.Q_sa[(curr_state, curr_action)] + self.alpha*sample
                curr_state = next_state
                if (self.sarsa == True):
                    curr_action = next_action
                else:
                    curr_action = self.getEpsilonAction(curr_state)
            self.step_num += 1
            iter+=1


    
    def train_model(self, analyze_performance=False):
        for _ in tqdm(range(self.max_episodes)):
            self.run_episode()
            if (analyze_performance == True):
                self.getPolicy()
                if (self.episode_num in self.episodes):
                    self.discounted_rewards.append(self.average_dis_reward())
        self.getPolicy()


    def getPolicy(self):
        for states in self.states.values():
            self.policy[states] = self.get_greedyAction(states)
