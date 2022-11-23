import numpy as np
import random
import math
from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# random.seed(1)

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
    
        if (col >=0 and col <=3) and (row == 0 or row ==3 or row ==7) and action == 'East':
            return (row, col)
        if (col >=6 and col <=9) and (row ==2 or row== 7) and action == 'East':
            return (row, col)
        if (col >=4 and col <= 7) and (row==5) and action == 'East':
            return (row, col)
        if (col >=0 and col <=3) and (row == 1 or row ==4 or row ==8) and action == 'West':
            return (row, col)
        if (col >=6 and col <=9) and (row ==3 or row== 8) and action == 'West':
            return (row, col)
        if (col >=4 and col <= 7) and (row==6) and action == 'West':
            return (row, col)
    
        if action == 'North':
            if (col + 1 >= self.COL_COUNT):
                return (row, col)
            return (row, col+1)
        elif action == 'East':
            if (row + 1 >= self.ROW_COUNT):
                return (row, col)
            return (row+1, col)
        elif action == 'South':
            if (col - 1 < 0):
                return (row, col)
            return (row, col-1)
        elif action == 'West':
            if (row - 1 < 0):
                return (row, col)
            return (row-1, col)



class MDPModel:
    def __init__(self, roadMap, alpha = 0.25, epsilon = 0.1, max_steps = 500, max_episodes = 5000, discount_factor = 0.99, decaying = False) -> None:
        self.roadMap = roadMap
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
        self.episodes = range(100,1000,100)
        self.discounted_rewards =  []
        self.final_policy_arr = []
        self.states_value_initialisation()
        


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
        if next_state.curr_x == self.roadMap.goal[0] and next_state.curr_y == self.roadMap.goal[1]:
            return +5
        elif next_state.curr_x == state.curr_x and next_state.curr_y == state.curr_y:
            return -0.02
        elif self.get_distance(next_state) < self.get_distance(state):
            return +0.8
        else:
            return -0.5


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
        
        iter=0
        while (iter < self.max_steps):
            action_taken = self.get_action(curr_action)
            next_state =  self.get_next_state(curr_state, action_taken)
            reward = self.get_reward(curr_state, curr_action, next_state)
            next_action = self.get_epsilonAction(next_state)

            if (curr_state == self.states[(self.roadMap.goal[0], self.roadMap.goal[1], self.roadMap.goal[0], self.roadMap.goal[1])]):
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
        # self.final_policy_arr.append((nxt.curr_x,nxt.curr_y))
        counter = 0
        while True:
            self.final_policy_arr.append((nxt.curr_x,nxt.curr_y))
            if (nxt.curr_x==nxt.goal_x and nxt.curr_y == nxt.goal_y) or counter==50:
                # self.final_policy_arr.append((nxt.curr_x,nxt.curr_y))
                break

            suggested_act = self.policy[(nxt.curr_x, nxt.curr_y,nxt.goal_x,nxt.goal_y)] 
            taken_act = self.get_action(suggested_act)
            next_state = self.get_next_state(nxt,taken_act)
            print(f"step={counter},{nxt}+{taken_act} --> {next_state}")
            nxt = next_state
            # self.final_policy_arr.append((nxt.curr_x,nxt.curr_y))
            counter+=1

    def gridAnimate(self,array):
        
        maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
        ]

        # array = [(4,0),(4,1),(4,2),(4,3),(5,3),(6,3),(6,4)]

        plt.rcParams["figure.figsize"] = [7.00, 10.00]
        plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()

        def update(t):
            nonlocal maze
            nonlocal array
            if t==len(array):
                return 
            if t==0:
                idx_x,idx_y = 9-array[t][1],array[t][0]
                
                maze[idx_x][idx_y]=10
                maze[9-array[-1][1]][array[-1][0]] = 20
            
            else:
                prev_idx_x,prev_idx_y = 9-array[t-1][1],array[t-1][0]
                idx_x,idx_y = 9-array[t][1],array[t][0]

                maze[prev_idx_x][prev_idx_y]=1
                maze[idx_x][idx_y]=10
                maze[9-array[-1][1]][array[-1][0]] = 20

            ax.imshow(maze,cmap="Greens")

        anim = ani.FuncAnimation(fig, update, frames=len(array), interval=1000)
        plt.show()



def analyze_discounted_rewards(roadMap, decaying, alpha = 0.25, epsilon = 0.1, discount_factor = 0.99, max_episodes = 5000):
    
    model = MDPModel(roadMap, decaying=decaying, alpha= alpha, epsilon=epsilon, discount_factor=discount_factor, max_episodes=max_episodes)
    np.random.seed(0)
    model.train_model(analyze_performance=True)
    model.simulate_policy(model.roadMap.person_init,model.roadMap.goal)
    model.gridAnimate(model.final_policy_arr)
    print(f"Final discount: {model.discounted_rewards[-1]}")




person_init = (4,0)
destination = (4,4)

roadMap = roadMap(person_init, destination)

analyze_discounted_rewards(roadMap, True, max_episodes=1000)


# general of 
# autism and IEEE transaction 
# general of autism