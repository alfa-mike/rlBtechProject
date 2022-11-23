from agentQL import *
from agentSA import *
from environment import *


def _callingFnQ(FrozenLake, decaying, alpha=0.25, epsilon=0.1, discount_factor=0.99, max_episodes=5000):
    agent = rlAgentQL(FrozenLake, decaying = decaying, alpha = alpha, epsilon = epsilon, discount_factor = discount_factor, max_episodes = max_episodes)
    np.random.seed(0)
    agent.train_agent(analyze_performance = True)
    agent.simulate_policy(agent.FrozenLake.person, agent.FrozenLake.goal)



def _callingFnS(FrozenLake, decaying, alpha=0.25, epsilon=0.1, discount_factor=0.99, max_episodes=5000):
    agent = rlAgentSA(FrozenLake, decaying = decaying, alpha = alpha, epsilon = epsilon, discount_factor = discount_factor, max_episodes = max_episodes)
    np.random.seed(0)
    agent.train_agent(analyze_performance = True)
    agent.simulate_policy(agent.FrozenLake.person, agent.FrozenLake.goal)


def main():
    person = (1,0)
    goal = (6,7)

    Frozenlake = Environment(person, goal)
    _callingFnQ(Frozenlake, True, max_episodes = 2000)
    # _callingFnS(Frozenlake, True, max_episodes = 2000)



if __name__ == "__main__":
    main()