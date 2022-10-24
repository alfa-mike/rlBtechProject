import numpy as np
import random
import math
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.animation as animations

# random.seed(1)

class State:
    def __init__(self, curr_x:int, curr_y:int, goal_x:int, goal_y:int )->None:
        self.curr_x = curr_x
        self.curr_y = curr_y
        self.goal_x = goal_x
        self.goal_y = goal_y

    def __str__(self) -> str:
        return f"({self.curr_x}, {self.curr_y}, {self.goal_x}, {self.goal_y})"


class frozenLake:
    ROW_COUNT = 10
    COL_COUNT = 10

    def __init__(self, person_init, goal)->None:
        self.person_init = person_init
        self.goal = goal
        #the map contains Holes and Obstructions, the holes represent the locations from which the agent cannnot recover
        self.holes = {(1,1), (1,2), (2,2), (2,6), (5,5), (6,8)}
    def next_cell(self, x, y, action):
        # print(x,y,action)
        assert x >= 0 and x < self.COL_COUNT, "x out of range"
        assert y >= 0 and y < self.ROW_COUNT, "y out of range"
        assert action in ['North', 'East', 'South', 'West', 'Down'], "invalid action"
        #if the person has fallen in the hole, then the person cannot move
        if action == "Down":
            return (x,y)
        if ( y >= 0 and y<= 3) and (x == 0 or x == 3 or x == 7) and action == 'East':
            return (x, y)
        if (y >= 6 and y<= 9) and (x == 2 or x == 7) and action == 'East':
            return (x, y)
        if (y >= 4 and y<= 7) and (x == 5) and action == 'East':
            return (x, y)
        if (y >= 0 and y<= 3) and (x == 1 or x == 4 or x == 8) and action == 'West':
            return (x, y)
        if (y >= 6 and y<= 9) and (x == 3 or x == 8) and action == 'West':
            return (x, y)
        if (y >= 4 and y<= 7) and (x == 6) and action == 'West':
            return (x, y)
        if action == 'North':
            if (y + 1 >= self.ROW_COUNT):
                return (x, y)
            return (x, y+1)
        elif action == 'East':
            if (x + 1 >= self.COL_COUNT):
                return (x, y)
            return (x+1, y)
        elif action == 'South':
            if (y - 1 < 0):
                return (x, y)
            return (x, y-1)
        elif action == 'West':
            if (x - 1 < 0):
                return (x, y)
            return (x-1, y)
        
    
    # def next_cell(self, row, col, action):
    #     assert row>=0 and row<self.ROW_COUNT, "row out of range" 
    #     assert col>=0 and col<self.COL_COUNT, "col out of range"
    #     assert action in ['North', 'East', 'South', 'West'], "invalid action"

    #     #obstructions in the map, hardcoding the obstructions/walls
    #     #here row is x and col is y
    #     if (col >=0 and col <=3) and (row == 0 or row ==3 or row ==7) and action == 'East':
    #         return (row, col)
    #     if (col >=6 and col <=9) and (row ==2 or row== 7) and action == 'East':
    #         return (row, col)
    #     if (col >=4 and col <= 7) and (row==5) and action == 'East':
    #         return (row, col)
    #     if (col >=0 and col <=3) and (row == 1 or row ==4 or row ==8) and action == 'West':
    #         return (row, col)
    #     if (col >=6 and col <=9) and (row ==3 or row== 8) and action == 'West':
    #         return (row, col)
    #     if (col >=4 and col <= 7) and (row==6) and action == 'West':
    #         return (row, col)
    
    #     if action == 'North':
    #         if (col + 1 >= self.COL_COUNT):
    #             return (row, col)
    #         return (row, col+1)
    #     elif action == 'East':
    #         if (row + 1 >= self.ROW_COUNT):
    #             return (row, col)
    #         return (row+1, col)
    #     elif action == 'South':
    #         if (col - 1 < 0):
    #             return (row, col)
    #         return (row, col-1)
    #     elif action == 'West':
    #         if (row - 1 < 0):
    #             return (row, col)
    #         return (row-1, col)



class MDPModel:
    def __init__(self, frozenLake, alpha = 0.25, epsilon = 0.1, max_steps = 500, max_episodes = 5000, discount_factor = 0.99, decaying = False) -> None:
        self.frozenLake = frozenLake
        self.epsilon = epsilon
        self.alpha = alpha
        self.decaying = decaying
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.discount_factor = discount_factor
        self.actions = ['North', 'East', 'South', 'West']
        self.states = dict()
        self.states_idx = dict()
        self.Q_sa = dict()
        self.policy = dict()
        self.step_num = 1
        self.episode_num = 0
        #change self.episodes to calcuuuuulate the discounted for that episode
        self.episodes = range(0,2000,100)
        self.discounted_rewards =  []
        self.states_value_initialisation()
        
        #Policy initialisation
        # for state in self.states.values():
        #     self.policy[state] = "North"


    def states_value_initialisation(self)->None:
        for x in range(self.frozenLake.COL_COUNT):
            for y in range(self.frozenLake.ROW_COUNT):
#                 for goal_row in range(self.frozenLake.ROW_COUNT):
#                     for goal_col in range(self.frozenLake.COL_COUNT):
#                         self.states[(row, col, goal_row, goal_col)] = State(row, col, goal_row, goal_col)
                  goal_x = self.frozenLake.goal[0]
                  goal_y = self.frozenLake.goal[1]
                  self.states[(x, y, goal_x, goal_y )] = State(x, y, goal_x, goal_y)
        for i,state in enumerate(self.states.values()):
            self.states_idx[state] = i
            for action in self.actions:
                self.Q_sa[(state,action)] = 0
        for state in self.states.values():
            self.Q_sa[(state, "Down")] = 0


    def get_distance(self, state:State)->float:
        return float(abs(abs(state.curr_x - state.goal_x)**2 + abs(state.curr_y - state.goal_y)**2)**0.5)


    def get_reward(self, state:State, action, next_state:State)->float:
        assert action in self.actions or action == 'Down', "invalid action" 
        if next_state.curr_x == self.frozenLake.goal[0] and next_state.curr_y == self.frozenLake.goal[1]:
            return +50
        elif next_state.curr_x == state.curr_x and next_state.curr_y == state.curr_y:
            return -5
        elif (next_state.curr_x, next_state.curr_y) in self.frozenLake.holes:
            return -100
        elif self.get_distance(next_state) < self.get_distance(state):
            return +5
        else:
            return -10


    def get_next_state(self, state:State, action)->State:
        next_coord = self.frozenLake.next_cell(state.curr_x, state.curr_y, action)
        if next_coord == (state.curr_x, state.curr_y):
            return state
        else:
            return self.states[(next_coord[0], next_coord[1], state.goal_x, state.goal_y)]


    def generate_random_state(self) -> State:
        return np.random.choice(list(self.states.values()))
    

    def get_action(self, action_suggested)->str:
        if action_suggested == "Down":
            return 'Down' 
        probs = [0.05, 0.05,0.05, 0.05]
        dir = ['North', 'East', 'South', 'West']
        probs[dir.index(action_suggested)] = 0.85
        return np.random.choice(dir, p=probs)


    def get_greedyAction(self, state:State) -> str:
        if (state.curr_x, state.curr_y) in self.frozenLake.holes:
            return "Down"
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
        if action == "Down":
            return action
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
        
        iter=0
        while (iter < self.max_steps):
            action_taken = self.get_action(curr_action)
            next_state =  self.get_next_state(curr_state, action_taken)
            reward = self.get_reward(curr_state, curr_action, next_state)
            next_action = self.get_epsilonAction(next_state)

            if (curr_state == self.states[(self.frozenLake.goal[0], self.frozenLake.goal[1], self.frozenLake.goal[0], self.frozenLake.goal[1])]):
                self.Q_sa[(curr_state, curr_action)] = reward 
                break
            elif (curr_state.curr_x, curr_state.curr_y) in self.frozenLake.holes:
                self.Q_sa[(curr_state, curr_action)] = reward 
                break
            else:
                sample= reward + self.discount_factor*self.Q_sa[(next_state, next_action)]
                self.Q_sa[(curr_state, curr_action)] = (1-self.alpha)*self.Q_sa[(curr_state, curr_action)] + self.alpha*sample
                curr_state = next_state
                curr_action = self.get_epsilonAction(curr_state)
                
            self.step_num += 1
            iter+=1


    def average_dis_reward(self, runs = 100):
        sum = 0
        for _ in range(runs):
            #change max_iters here
            sum += self.simulate(max_iters=500,  verbose=True)
        return sum/runs

    
    def getPolicy(self):
        for states in self.states.values():
            self.policy[(states.curr_x,states.curr_y,states.goal_x,states.goal_y)] = self.get_greedyAction(states)



    def simulate(self, max_iters = 500, verbose =  False):
        
        curr_state = random.choice(list(self.states.values()))
        discounted_reward = 0
        i=0
        while(i <max_iters):
            action_suggested = self.policy[(curr_state.curr_x,curr_state.curr_y,curr_state.goal_x,curr_state.goal_y)]
            action_taken = self.get_action(action_suggested)
            next_state = self.get_next_state(curr_state, action_taken)
            discounted_reward += self.get_reward(curr_state, action_taken, next_state)*(self.discount_factor**i)
            # if (verbose ==  True):
            #     print(f"{i+1}: {curr_state} -> {action_suggested} : {action_taken} -> {next_state}")
            if next_state.curr_x == next_state.goal_x and next_state.curr_y == next_state.goal_y:
                break
            curr_state = next_state
            i+=1
        return discounted_reward


    def train_model(self, analyze_performance=False):
        for _ in tqdm(range(self.max_episodes)):
            self.run_episode()
            if (analyze_performance == True):
                self.getPolicy()
                if (self.episode_num in self.episodes):
                    self.discounted_rewards.append(self.average_dis_reward())
        self.getPolicy()

    
    def plot_dis_rewards(self):
        # make episodes as x axis and discounted rewards as y axis
        plt.plot(self.episodes, self.discounted_rewards)
        plt.title(f"Figure : alpha:{self.alpha}, epsilon:{self.epsilon}")
        plt.xlabel("Episodes")
        plt.ylabel("Discounted Rewards")
        plt.savefig(f"outputs/alpha:{self.alpha}_epsilon:{self.epsilon}.png")
        plt.show()
        plt.close()



    def simulate_policy(self,init,dest):
        nxt = State(init[0],init[1],dest[0],dest[1])
        final = State(dest[0],dest[1],dest[0],dest[1])
        print(nxt,final)
        counter = 0
        while True:
            
            if (nxt.curr_x==nxt.goal_x and nxt.curr_y == nxt.goal_y) or counter==50:
                # print("in")
                break
            
            # print(nxt)
            suggested_act = self.policy[(nxt.curr_x, nxt.curr_y,nxt.goal_x,nxt.goal_y)] 
            taken_act = self.get_action(suggested_act)
            next_state = self.get_next_state(nxt,taken_act)
            print(counter,nxt,taken_act,next_state)
            nxt = next_state
            counter+=1


def analyze_discounted_rewards(frozenLake, decaying, alpha = 0.25, epsilon = 0.1, discount_factor = 0.99, max_episodes = 5000):
    
    model = MDPModel(frozenLake, decaying=decaying, alpha= alpha, epsilon=epsilon, discount_factor=discount_factor, max_episodes=max_episodes)
    np.random.seed(0)
    model.train_model(analyze_performance=True)
    for k,v in model.policy.items():
        print(k,v)
    model.simulate_policy(model.frozenLake.person_init,model.frozenLake.goal)
    # model.plot_dis_rewards()
    print(f"Final discount: {model.discounted_rewards[-1]}")




person_init = (2,9)
destination = (3,6)

frozenLake = frozenLake(person_init, destination)

analyze_discounted_rewards(frozenLake, True, max_episodes=4000)
