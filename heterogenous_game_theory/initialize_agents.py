import pandas as pd
import numpy as np
import scipy as sp
from .agent import Agent
from scipy.stats import truncnorm
from itertools import combinations
import matplotlib.pyplot as plt

def get_agents(homogenous = True, number_of_agents = 100, M = [], D = [], R = [], W = []):

    """
    This function creates a set of agents of size number_of_agents. To allow for maxmimum customization,
    meaning that each characteric of the agent can follow a unique distribution, the user must specify
    if the distribution of an element should follow a power-law by making the first element of M, D, and/or R
    'power'. Otherwise, the element will follow a normal distribution. 
    """

    if homogenous:
        # if homogenous we create four columns with default values
        agents_test = ["Agent " + str(i) for i in range(number_of_agents)]
        M = [5 for i in range(number_of_agents)]
        D = [0.4 for i in range(number_of_agents)]
        R = [0.3 for i in range(number_of_agents)]
        W = [1 for i in range(number_of_agents)]

        data = { "Agents": agents_test, "M" : M, "D" : D, "R" : R, "W": W} 
        df = pd.DataFrame.from_dict(data)
        
        agents = []

        for name in df.index:
            agent = Agent(name, df['M'].at[name], df['D'].at[name], df['R'].at[name],df['W'].at[name] )
            agents.append(agent)
        
        return agents

    # if not homogenous we need to take into account different combinations of distributions. 
    if M[0] == "power":
        x_m, alpha_m = M[1], M[2]
        # from numpy documentation: The classical Pareto distribution can be obtained from the Lomax distribution by adding 1 and multiplying by the scale parameter m.
        samples_m = (np.random.pareto(alpha_m, 100000) + 1) * x_m
        samples_ints_m = [round(i,1) for i in samples_m]
        samples_ints_m = [i for i in samples_ints_m if i <= 10]
        M = [np.random.choice(samples_ints_m) for i in range(number_of_agents)]
    else:
        # if the distribution does not follow a power law, we take a truncated normal distribution
        M_trunc = get_truncated_normal(mean=M[0], sd = M[1], low = 0, upp = 10)
        M = [round(np.random.choice([i for i in M_trunc.rvs(1000)]), 1) for i in range(number_of_agents)]
        
    # we repeat this for the other variables 
    if D[0] == "power":
        x_d, alpha_d = D[1], D[2]
        samples_d = (np.random.pareto(alpha_d, 100000) + 1) * x_d
        samples_ints_d = [round(i,2) for i in samples_d]
        samples_ints_d = [i for i in samples_ints_d if i <= 1]
        D = [np.random.choice(samples_ints_d) for i in range(number_of_agents)]
    else:
        D_trunc = get_truncated_normal(mean=D[0], sd = D[1], low = 0.01, upp = 1)
        D = [round(np.random.choice([i for i in D_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]
    
    if R[0] == "power":
        x_r, alpha_r = R[1], R[2]
        samples_r = (np.random.pareto(alpha_r, 100000) + 1) * x_r
        samples_ints_r = [round(i,2) for i in samples_r]
        samples_ints_r = [i for i in samples_ints_r if i <= 1]
        R = [np.random.choice(samples_ints_r) for i in range(number_of_agents)]
    else:
        R_trunc = get_truncated_normal(mean=R[0], sd = R[1], low = 0.01, upp = 1)
        R = [round(np.random.choice([i for i in R_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]


    if W[0] == "power":
        x_r, alpha_r = W[1], W[2]
        samples_r = (np.random.pareto(alpha_r, 100000) + 1) * x_r
        samples_ints_r = [round(i,2) for i in samples_r]
        samples_ints_r = [i for i in samples_ints_r if i <= 1]
        W = [np.random.choice(samples_ints_r) for i in range(number_of_agents)]
    else:
        W_trunc = get_truncated_normal(mean=W[0], sd = W[1], low = 0, upp = 1)
        W = [round(np.random.choice([i for i in W_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]

    agents_test = ["Agent " + str(i) for i in range(number_of_agents)]
    data = {"Agents": agents_test, "M" : M, "D" : D, "R" : R, "W" : W}
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
