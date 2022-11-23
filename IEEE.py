import numpy as np
import math
import random
from tqdm import *
import matplotlib.pyplot as plt
from matplotlib import animation


class State:
    def __init__(self, curr_x, curr_y, goal_x, goal_y) -> None:
        self.curr_x = curr_x
        self.curr_y = curr_y
        self.goal_x = goal_x
        self.goal_y = goal_y

    def __str__(self) -> str:
        return f"({self.curr_x},{self.curr_y},{self.goal_x},{self.goal_y})"

class FrozenLake:
    ROW_COUNT = 10
    COL_COUNT = 10

    def __init__(self, person, goal) -> None:
        self.person = person
        self.goal = goal
        self.holes = {(1,1), (1,2), (2,2), (2,6), (5,5), (6,8)}

    def next_cell(self, x,y,action):
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

class IEEEAgent:
    def __init__(self, FrozenLake, alpha = 0.25, epsilon = 0.1, maxsteps = 500, max_episodes = 5000, discount_factor = 0.99, decaying = False) -> None:
        self.FrozenLake = FrozenLake
        self.alpha = alpha
        self.epsilon = epsilon
        self.maxsteps = maxsteps
        self.max_episodes = max_episodes
        self.discount_factor = discount_factor
        self.decaying = decaying
        self.actions = ['North', 'East', 'South', 'West']
        self.states = dict()
        self.states_idx = dict()
        self.Q_sa = dict()
        self.policy = dict()
        self.step_num = 1
        self.episode_num = 0
        #change self.episodes to calcuuuuulate the discounted for that episode
        self.episodes = range(0,2000,5)
        self.discounted_rewards =  []
        self.normal_rewards = []
        self.states_value_initialisation()
        
    
    def states_value_initialisation(self)->None:
        for x in range(self.FrozenLake.COL_COUNT):
            for y in range(self.FrozenLake.ROW_COUNT):

                # for goal_row in range(self.frozenLake.ROW_COUNT):
                #     for goal_col in range(self.frozenLake.COL_COUNT):
                #         self.states[(row, col, goal_row, goal_col)] = State(row, col, goal_row, goal_col)
                goal_x = self.FrozenLake.goal[0]
                goal_y = self.FrozenLake.goal[1]
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
        if next_state.curr_x == self.FrozenLake.goal[0] and next_state.curr_y == self.FrozenLake.goal[1]:
            return +50
        elif next_state.curr_x == state.curr_x and next_state.curr_y == state.curr_y:
            return -5
        elif (next_state.curr_x, next_state.curr_y) in self.FrozenLake.holes:
            return -60
        elif self.get_distance(next_state) < self.get_distance(state):
            return +5
        else:
            return -10
   
    def get_next_state(self, state:State, action)->State:
        next_coord = self.FrozenLake.next_cell(state.curr_x, state.curr_y, action)
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
        if (state.curr_x, state.curr_y) in self.FrozenLake.holes:
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
        self.episode_num += 1
        curr_state = self.generate_random_state()
        curr_action = self.get_epsilonAction(curr_state)
        iter = 0
        while iter < self.maxsteps:
            action_taken = self.get_action(curr_action)
            next_state = self.get_next_state(curr_state, action_taken)
            reward = self.get_reward(curr_state, curr_action, next_state)
            next_action = self.get_greedyAction(next_state)

            if (curr_state == self.states[(self.FrozenLake.goal[0], self.FrozenLake.goal[1], self.FrozenLake.goal[0], self.FrozenLake.goal[1])]):
                self.Q_sa[(curr_state, curr_action)] = reward
                break
            elif(curr_state.curr_x, curr_state.curr_y) in self.FrozenLake.holes:
                self.Q_sa[(curr_state, curr_action)] = reward
                break
            else:
                sample = reward + self.discount_factor*self.Q_sa[(next_state, next_action)]
                self.Q_sa[(curr_state, curr_action)] = (1-self.alpha)*self.Q_sa[(curr_state, curr_action)] + self.alpha*sample
                curr_state = next_state
                curr_action = self.get_epsilonAction(curr_state)
            self.step_num += 1
            iter +=1
    
    def checkConvergence(self,prevP):
        if len(prevP.keys())<100:
            return False
        for k,v in self.policy.items():
            if prevP[k]!=v:
                return False
        return True


    def train_agent(self, analyze_performance = False):
        for _ in tqdm(range(int(self.max_episodes))):
            self.run_episode()
            if (analyze_performance == True):
                prevPolicy = self.policy.copy()
                self.getPolicy()
                if (self.episode_num % 5 == 0):
                    rewardN, rewardD = self.average_dis_reward()
                    self.normal_rewards.append(rewardN)
                    self.discounted_rewards.append(rewardD)
                    # if self.checkConvergence(prevPolicy):
                    #     break

        self.getPolicy()
        self.plot_dis_rewards()
        
    def plot_dis_rewards(self):
        # make episodes as x axis and discounted rewards as y axis
        print(f"episodes: {len(self.episodes)}")
        plt.plot(self.discounted_rewards)
        # plt.title(f"Figure : alpha:{self.alpha}, epsilon:{self.epsilon}")
        plt.xlabel("Episodes")
        plt.ylabel("Discounted Rewards")
        # plt.savefig(f"outputs/alpha:{self.alpha}_epsilon:{self.epsilon}.png")
        plt.show()
        plt.close()

    def average_dis_reward(self,runs = 100):
        sumN, sumD = 0, 0
        for _ in range(runs):
            #change max_iters here
            a,b = self.calc_disc_rewards(max_iters = 120, verbose = True)
            sumN += a
            sumD += b
        return round(sumN/runs,3), round(sumD/runs,3)
    
    def getPolicy(self):
        for states in self.states.values():
            self.policy[(states.curr_x, states.curr_y, states.goal_x, states.goal_y)] = self.get_greedyAction(states)

    def calc_disc_rewards(self, max_iters = 500, verbose = False):
        curr_state = random.choice(list(self.states.values()))
        discounted_reward = 0
        normal_reward = 0
        i=0
        while(i <max_iters):
            action_suggested = self.policy[(curr_state.curr_x,curr_state.curr_y,curr_state.goal_x,curr_state.goal_y)]
            action_taken = self.get_action(action_suggested)
            next_state = self.get_next_state(curr_state, action_taken)
            normal_reward += self.get_reward(curr_state, action_suggested, next_state)
            discounted_reward += self.get_reward(curr_state, action_taken, next_state)*(self.discount_factor**i)
            # if (verbose ==  True):
            #     print(f"{i+1}: {curr_state} -> {action_suggested} : {action_taken} -> {next_state}")
            if next_state.curr_x == next_state.goal_x and next_state.curr_y == next_state.goal_y:
                break
            if (next_state.curr_x, next_state.curr_y) in self.FrozenLake.holes:
                break
            curr_state = next_state
            i+=1
        return normal_reward, discounted_reward
    
    
    def simulate_policy(self, init, dest):
        
        fig = plt.figure("Simulator")
        fig.set_dpi(100)
        fig.set_size_inches(8,8)
        ax = plt.axes(xlim = (-0.5,10.5), ylim = (-1.5, 10.5))
        ax.set_facecolor("pink")
        plt.xticks([])
        plt.yticks([])


        line = plt.Line2D((0, 10), (0, 0), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((0, 10), (10, 10), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((0, 0), (0, 10), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((10, 10), (0, 10), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((1, 1), (0, 4), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((4, 4), (0, 4), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((8, 8), (0, 4), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((3, 3), (6, 10), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((8, 8), (6, 10), lw=2.5)
        plt.gca().add_line(line)
        line = plt.Line2D((6, 6), (4, 8), lw=2.5)
        plt.gca().add_line(line)
        

        for i in range(1,10):
            line = plt.Line2D((0, 10), (i, i), lw=1)
            plt.gca().add_line(line)
            line = plt.Line2D((i,i), (0,10), lw = 1)
            plt.gca().add_line(line)
        patch = plt.Rectangle((0.25, 0.25), 0.5, 0.5, fc = 'black')
        destcircle = plt.Circle((0.5,0.5), radius = 0.25, fc = 'blue')
        srccircle = plt.Circle((0.5,0.5), radius = 0.25, fc = 'green')

        for hole in self.FrozenLake.holes:
            reddp = plt.Circle((0.5, 0.5), radius=0.25, fc='red')
            ax.add_patch(reddp)
            reddp.center = (hole[0] + 0.5, hole[1]+ 0.5)
            plt.text(hole[0] + 0.35, hole[1]+0.35, "H", fontsize = 15)

        
        ax.add_patch(destcircle)
        ax.add_patch(patch)
        ax.add_patch(srccircle)
        state = State(init[0], init[1], dest[0], dest[1])
        patch.xy = (state.curr_x + 0.25, state.curr_y+ 0.25)
        destcircle.center = (state.goal_x + 0.5, state.goal_y + 0.5)
        srccircle.center = (state.curr_x + 0.5, state.curr_y + 0.5)
        plt.text(state.goal_x + 0.35, state.goal_y + 0.35, "G", fontsize = 15)
        plt.text(state.curr_x + 0.35, state.curr_y + 0.35, "S", fontsize = 15)
        def North(t,i):
            return (t[0]-0.5,t[1]+i-0.5)
        def South(t,i):
            return (t[0]-0.5,t[1]-i-0.5)
        def East(t,i):
            return (t[0]+i-0.5,t[1]-0.5)
        def West(t,i):
            return (t[0]-i-0.5,t[1]-0.5)
        def Down(t,i):
            return (t[0]-0.5,t[1]-0.5)
        def move(functp):
            def Fi(i):
                temp = functp(patch.xy, i)
                patch.xy = (temp[0]+0.5, temp[1]+0.5)
                return []
            return Fi
        # print('Starting simulation')
        helper_dict = {'North': North, 'South': South, 'East': East, 'West': West, 'Down': Down}
        plt.show(block = False)
        # return fig


        nxt = State(init[0],init[1],dest[0],dest[1])
        final = State(dest[0],dest[1],dest[0],dest[1])
        print(f"starting: {nxt} \t final: {final}")
        normal_reward, discounted_reward = 0,0
        text1 = plt.text(0, -1, f"Normal Reward: {normal_reward}", fontsize = 15)
        text2 = plt.text(5, -1, f"Discounted Reward: {discounted_reward}", fontsize = 15)
        counter = 0
        while True:
            #display current rewards on the figure
            # print(f"normal reward: {normal_reward} \t discounted reward: {discounted_reward}")
            text1.set_text(f"Normal Reward: {normal_reward}")
            text2.set_text(f"Discounted Reward: {round(discounted_reward,3)}")

            if (nxt.curr_x == nxt.goal_x and nxt.curr_y == nxt.goal_y) or counter == 30:
                break
            if (nxt.curr_x, nxt.curr_y) in self.FrozenLake.holes:
                break
            suggested_act = self.policy[(nxt.curr_x, nxt.curr_y, nxt.goal_x, nxt.goal_y)]
            taken_action = suggested_act # self.get_action(suggested_act)
            next_state = self.get_next_state(nxt, taken_action)
            print(counter, (nxt.curr_x, nxt.curr_y), taken_action, (next_state.curr_x, next_state.curr_y))

            normal_reward += self.get_reward(nxt, taken_action, next_state)
            discounted_reward += self.get_reward(nxt, taken_action, next_state) * self.discount_factor**counter
            if next_state == nxt:
                anim = animation.FuncAnimation(fig, move(helper_dict["Down"]), frames=2, interval=10, blit = True, repeat = False)
            else:
                anim = animation.FuncAnimation(fig, move(helper_dict[taken_action]), frames = 2, interval = 10, blit = True, repeat = False)
            plt.show(block = False)
            plt.pause(2)
            nxt = next_state
            counter += 1

        print(f"Normal Reward: {normal_reward} \t Discounted Reward: {discounted_reward}")
        if (nxt.curr_x, nxt.curr_y) in self.FrozenLake.holes:
            print("Episode Failed")
            plt.text(2.5 + 0.35, 4 + 0.35, "Episode Failed", fontsize = 30, bbox = dict(facecolor = 'red', alpha = 0.5))
        else:
            if counter == 30:
                print("Episode terminated due to time limit")
                plt.text(0 + 0.35, 4 + 0.35, "Episode terminated due to time limit", fontsize = 22, bbox = dict(facecolor = 'red', alpha = 0.5))
            else:
                print("Episode successful: Reached Goal")
                plt.text(0 + 0.35, 4 + 0.35, "Episode Successful: Reached Goal", fontsize = 22, bbox = dict(facecolor = 'green', alpha = 0.5))
        plt.pause(2)
        plt.close()


def analyze_discounted_rewards(FrozenLake, decaying, alpha = 0.25,epsilon= 0.1, discount_factor = 0.99, max_episodes = 5000):

    agent = IEEEAgent(FrozenLake, decaying = decaying, alpha = alpha, epsilon = epsilon, discount_factor = discount_factor, max_episodes = max_episodes)
    np.random.seed(0)
    agent.train_agent(analyze_performance = True)
    agent.simulate_policy(agent.FrozenLake.person, agent.FrozenLake.goal)

    # print(f"final discount: {agent.discounted_rewards[-1]}")
def main():
    person = (1,0)
    goal = (6,7)

    Frozenlake = FrozenLake(person, goal)
    analyze_discounted_rewards(Frozenlake, True, max_episodes = 2000)

if __name__ == "__main__":
    main()