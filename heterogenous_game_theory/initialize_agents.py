import pandas as pd
import numpy as np
import scipy as sp
from .agent import Agent
from .payoff_functions import game_types
from scipy.stats import truncnorm
from itertools import combinations
import matplotlib.pyplot as plt

def get_agents(number_of_agents = 100, game_type = None, Weight = [1, 1/10000]):

    """
    This function creates a set of agents of size number_of_agents. To allow for maxmimum customization,
    meaning that each characteric of the agent can follow a unique distribution, the user must specify
    if the distribution of an element should follow a power-law by making the first element of M, D, and/or R
    'power'. Otherwise, the element will follow a normal distribution. Custom specifications can be added to the
    game_types dictionary. 
    """
    # check if game type has been specified, otherwise return false.
    if not game_type:
        print("Please specify game type.", "Choose from the following: ")
        for key, value in game_types.items():
            print(key)
        print("Or create your own.")

        return False
    
    # check if game type exists in dictionary, otherwise return false.
    if game_type not in game_types.keys():
        print("This game type is not included. Please append it to the game_types dictionary and try again.")
        return False

    # retrieve the three values that characterize the agents.
    Mass = game_types[game_type]['M']
    Dependence = game_types[game_type]['D']
    Rivalry = game_types[game_type]['R']

    # check if distribution is a pareto and create agents, otherwise create
    # agents based on normal distributions with mean and s.d..
    if Mass[0] == "power":
        x_m, alpha_m = Mass[1], Mass[2]
        # numpy documentation: The classical Pareto distribution can be obtained
        # from the Lomax distribution by adding 1 and multiplying by the scale parameter m.
        pareto_distribution = (np.random.pareto(alpha_m, 100000) + 1) * x_m
        pareto_sample = [round(i,1) for i in pareto_distribution]
        pareto_sample_truncated = [i for i in pareto_sample if i <= 10]
        M = [np.random.choice(pareto_sample_truncated) for i in range(number_of_agents)]
    else:
        # if the distribution does not follow a power law, we take a truncated normal distribution
        M_trunc = get_truncated_normal(mean=Mass[0], sd = Mass[1], low = 0.1, upp = 10)
        M = [round(np.random.choice([i for i in M_trunc.rvs(1000)]), 1) for i in range(number_of_agents)]
        
    # we repeat this for the other variables 
    if Dependence[0] == "power":
        x_d, alpha_d = Dependence[1], Dependence[2]
        pareto_distribution = (np.random.pareto(alpha_d, 100000) + 1) * x_d
        pareto_sample = [round(i,2) for i in pareto_distribution]
        pareto_sample_truncated = [i for i in pareto_sample if i <= 1]
        D = [np.random.choice(pareto_sample_truncated) for i in range(number_of_agents)]
    else:
        D_trunc = get_truncated_normal(mean=Dependence[0], sd = Dependence[1], low = 0.01, upp = 1)
        D = [round(np.random.choice([i for i in D_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]
    
    if Rivalry[0] == "power":
        x_r, alpha_r = Rivalry[1], Rivalry[2]
        pareto_distribution = (np.random.pareto(alpha_r, 100000) + 1) * x_r
        pareto_sample = [round(i,2) for i in pareto_distribution]
        pareto_sample_truncated = [i for i in pareto_sample if i <= 1]
        R = [np.random.choice(pareto_sample_truncated) for i in range(number_of_agents)]
    else:
        R_trunc = get_truncated_normal(mean=Rivalry[0], sd = Rivalry[1], low = 0.01, upp = 1)
        R = [round(np.random.choice([i for i in R_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]

    if Weight[0] == "power":
        x_r, alpha_r = Weight[1], Weight[2]
        pareto_distribution = (np.random.pareto(alpha_r, 100000) + 1) * x_r
        pareto_sample = [round(i,2) for i in pareto_distribution]
        pareto_sample_truncated = [i for i in pareto_sample if i <= 1]
        W = [np.random.choice(pareto_sample_truncated) for i in range(number_of_agents)]
    else:
        W_trunc = get_truncated_normal(mean=Weight[0], sd = Weight[1], low = 0, upp = 1)
        W = [round(np.random.choice([i for i in W_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]

    agents_index = ["Agent " + str(i) for i in range(number_of_agents)]
    data = {"Agents": agents_index, "M" : M, "D" : D, "R" : R, "W" : W}
    df = pd.DataFrame.from_dict(data)

    # now we create the agent objects
    agents = []

    for name in df.index:
        agent = Agent(name, df['M'].at[name], df['D'].at[name], df['R'].at[name], df['W'].at[name])
        agents.append(agent)

    return agents

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def check_parameters(agents):
    
    sdm = round(np.std([agent.m for agent in agents]),2)
    sdi = round(np.std([agent.r for agent in agents]),2)
    sde = round(np.std([agent.d for agent in agents]),2)
    mm = round(np.mean([agent.m for agent in agents]),2)
    mi = round(np.mean([agent.r for agent in agents]),2)
    me = round(np.mean([agent.d for agent in agents]),2)
    
    print("AGENT PARAMETERS IN POPULATION")
    print(37 * "-")
    print("   M \t\t  D \t\t  R")
    print(37 * "-")
    
    for agent in agents[0:19]:
        print("|", agent.m, "\t\t", agent.d, "\t\t", agent.r, "|")
        
    print(37 * "-")
    
    print(f"The s.d. of M is: {sdm}")
    print(f"The s.d. of D is: {sdi}")
    print(f"The s.d. of R is: {sde}")
    
    print(37 * "-")
    
    print(f"The mean of M is: {mm}")
    print(f"The mean of D is: {mi}")
    print(f"The mean of R is: {me}")
    
    print(37 * "-")

    #subplot with 1 row & 3 cols
    fig, ax = plt.subplots(1,3, figsize = (30,10))
    ax[0].hist([agent.m for agent in agents])
    ax[1].hist([agent.d for agent in agents])
    ax[2].hist([agent.r for agent in agents])
    ax[0].set_title('Agent M')
    ax[1].set_title('Agent d')
    ax[2].set_title('Agent r')
    plt.show()

def compare_payoff_function(agents, payoff_functions):
        """      
        parameters:
            - countries: list of Country, countries that take part in the Tournament
        """            
        # loop through all combinations of countries in the tournament.
        # the index of country 1 (c1) is always less than that of country 2 (c2)
        for c1, c2 in combinations(agents[0:5], 2):
            # calculate the payoff values that are associated with games 
            # between these two countries, and store them as tupples
            # R: reward, P: punishment, T: temptation, S: sucker
            RR = (payoff_functions['R'](c1,c2), 
                  payoff_functions['R'](c2,c1))
            
            PP = (payoff_functions['P'](c1,c2), 
                  payoff_functions['P'](c2,c1))
            
            TS = (payoff_functions['T'](c1,c2), 
                  payoff_functions['S'](c2,c1))
            
            ST = (payoff_functions['S'](c1,c2), 
                  payoff_functions['T'](c2,c1))
            
            print(100 * "_")
            print(f"Agent {c1.name} playing Agent {c2.name}: Reward: {RR[0]}, Temptation: {TS[0]}, Sucker: {ST[0]}, Punishment: {PP[0]}")
            print(f"Agent {c2.name} playing Agent {c1.name}: Reward: {RR[1]}, Temptation: {ST[1]}, Sucker: {TS[1]}, Punishment: {PP[1]}")

        R = []
        T = []
        P = []
        S = []
        for c1, c2 in combinations(agents, 2):
            # calculate the payoff values that are associated with games 
            # between these two countries, and store them as tupples
            # R: reward, P: punishment, T: temptation, S: sucker
            RR = (payoff_functions['R'](c1,c2), 
                payoff_functions['R'](c2,c1))

            PP = (payoff_functions['P'](c1,c2), 
                payoff_functions['P'](c2,c1))

            TS = (payoff_functions['T'](c1,c2), 
                payoff_functions['S'](c2,c1))

            ST = (payoff_functions['S'](c1,c2), 
                payoff_functions['T'](c2,c1))

            R.append(RR[0])
            T.append(TS[0])
            S.append(ST[0])
            P.append(PP[0])
            R.append(RR[1])
            T.append(ST[1])
            S.append(TS[1])
            P.append(PP[1])

        for i in range(len(R)):
            if T[i] >= R[i] >= P[i] >= S[i]:
                continue
            else:
                print(i)
                print("NOT ALL PLAYERS ARE PLAYING A PRISONER'S DILEMMA")
                return False

        print("\nAll players are playing a prisoners dilemma")   
        return 0          