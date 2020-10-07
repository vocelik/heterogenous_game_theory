import pandas as pd
import numpy as np
import scipy as sp
from .agent import Agent
from scipy.stats import truncnorm
from itertools import combinations
import matplotlib.pyplot as plt

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_agents(homogenous = True, power = False, number_of_agents = 100, M = [], E = [], I = []):

    """
    This function creates a set of agents of size number_of_agents. To allow for maxmimum customization,
    meaning that each characteric of the agent can follow a unique distribution, the user must specify
    if distribution of an element should follow a power-law by making the first element of M, E, and/or I
    'power'. Otherwise, the element will follow a normal distribution. 
    """

    agents_test = ["Agent " + str(i) for i in range(number_of_agents)]

    if homogenous:
        # if homogenous we create four columns with default values
        agents_test = ["Agent " + str(i) for i in range(number_of_agents)]
        M = [10 for i in range(number_of_agents)]
        E = [0.3 for i in range(number_of_agents)]
        I = [0.3 for i in range(number_of_agents)]

        data = { "Agents": agents_test, "M" : M, "E" : E, "I" : I}
        df = pd.DataFrame.from_dict(data)
        
        agents = []
        for name in df.index:
            agent = Agent(name, df['M'].at[name], df['E'].at[name], df['I'].at[name])
            agents.append(agent)
        
        return agents

    if M[0] == "power":
        x_m, alpha_m = M[1], M[2]
        samples_m = (np.random.pareto(alpha_m, 1000) + 1) * x_m
        samples_ints_m = [int(round(i)) for i in samples_m]
        M = [np.random.choice(samples_ints_m) for i in range(number_of_agents)]
    else:
        M_trunc = get_truncated_normal(mean=M[0], sd = M[1], low = 1, upp = M[0] + 10 * M[1])
        M = [round(np.random.choice([round(i) for i in M_trunc.rvs(1000)])) for i in range(number_of_agents)]

    if E[0] == "power":
        x_e, alpha_e = E[1], E[2]
        samples_e = (np.random.pareto(alpha_e, 1000) + 1) * x_e
        samples_ints_e = [round(i,2) for i in samples_e]
        E = [np.random.choice(samples_ints_e) for i in range(number_of_agents)]
        E = [v if v < 1.0 else 1 for v in E] # the max value E can take is 1
    else:
        E_trunc = get_truncated_normal(mean=E[0], sd = E[1], low = 0, upp = 1)
        E = [round(np.random.choice([round(i, 2) for i in E_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]
    
    if I[0] == "power":
        x_i, alpha__i = I[1], I[2]
        samples_i = (np.random.pareto(alpha__i, 1000) + 1) * x_i
        samples_ints_i = [round(i,2) for i in samples_i]
        I = [np.random.choice(samples_ints_i) for i in range(number_of_agents)]
        I = [v if v < 1.0 else 1 for v in I]
    else:
        I_trunc = get_truncated_normal(mean=I[0], sd = I[1], low = 0, upp = 1)
        I = [round(np.random.choice([round(i, 2) for i in I_trunc.rvs(1000)]), 2) for i in range(number_of_agents)]

        
    data = { "Agents": agents_test, "M" : M, "E" : E, "I" : I}
    df = pd.DataFrame.from_dict(data)

    agents = []
    for name in df.index:
        agent = Agent(name, df['M'].at[name], df['E'].at[name], df['I'].at[name])
        agents.append(agent)

    return agents

def check_parameters(agents):
    
    sdm = round(np.std([agent.m for agent in agents]))
    sdi =round(np.std([agent.i for agent in agents]),2)
    sde = round(np.std([agent.e for agent in agents]),2)
    mm = int(round(np.mean([agent.m for agent in agents])))
    mi = round(np.mean([agent.i for agent in agents]),2)
    me = round(np.mean([agent.e for agent in agents]),2)
    
    print("AGENT PARAMETERS IN POPULATION")
    print(37 * "-")
    print("   M \t\t  E \t\t  I")
    print(37 * "-")
    
    for agent in agents[0:19]:
        print("|", int(round(agent.m)), "\t\t", agent.e, "\t\t", agent.i, "|")
        
    print(37 * "-")
    
    print(f"The s.d. of M is: {sdm}")
    print(f"The s.d. of E is: {sdi}")
    print(f"The s.d. of I is: {sde}")
    
    print(37 * "-")
    
    print(f"The mean of M is: {mm}")
    print(f"The mean of E is: {mi}")
    print(f"The mean of I is: {me}")
    
    print(37 * "-")

    #subplot with 1 row & 3 cols
    fig, ax = plt.subplots(1,3, figsize = (30,10))
    ax[0].hist([agent.m for agent in agents])
    ax[1].hist([agent.e for agent in agents])
    ax[2].hist([agent.i for agent in agents])
    ax[0].set_title('Agent M')
    ax[1].set_title('Agent e')
    ax[2].set_title('Agent i')
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
            print(100 * "_")