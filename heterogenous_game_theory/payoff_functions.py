# These functions are the suggested payoff_functions. That is, they 
# calculate how much fitness a agent should  get/lose by
# playing a prisoners dilemma with another agent. They are
# the default payoff functions, but in the Tournament instance you
# could specify to use other functions.

def selfreward(agent):
    '''calculates how much reward a agent gets from its own internal market'''
    return agent.m

def reward(agent, other):
    '''calculates how much reward would change fitness'''
    return round((agent.d + agent.r) * other.m + (agent.w * agent.m), 2)

def temptation(agent, other):
    '''calculates how much temptation would change fitness'''
    return round((agent.d + 2 * agent.r) * other.m + (agent.w * agent.m), 2)

def sucker(agent, other):
    '''calculates how much sucker would change fitness'''
    return round(agent.w * agent.m, 2)

def punishment(agent, other):
    '''calculates how much punishment would chage fitness'''
    return round(agent.r * other.m + agent.w * agent.m, 2)

default_payoff_functions = {
        'R': reward,
        'T': temptation,
        'S': sucker,
        'P': punishment,
        'self_reward': selfreward
        }
        
traditional_payoff_functions = {
        'R': lambda *args: 3,
        'T': lambda *args: 5,
        'S': lambda *args: 0,
        'P': lambda *args: 1
        }

game_types = {
    
    "norm_mdr_max": {'M': [5, 12.5],
                        'D': [0.4, 1.25],
                        'R': [0.3, 1.25]},
    
    "norm_dr_max": {'M': [5, 1/10000],
                       'D': [0.4, 1.25],
                       'R': [0.3, 1.25]},
    
    "norm_m_max": {'M': [5, 12.5],
                   'D': [0.4, 1/10000],
                   'R': [0.3, 1/10000]},
    
    "pareto_mdr_max": {'M': ["power", 2.5, 1],
                       'D': ["power", 0.15, 1],
                       'R': ["power", 0.2, 1]},
    
    "pareto_dr_max": {'M': [5, 1/10000],
                      'D': ["power", 0.15, 1],
                      'R': ["power", 0.2, 1]},
    
    "pareto_m_max": {'M': ["power", 2.5, 1],
                     'D': [0.4, 1/10000],
                     'R': [0.3, 1/10000]},
    
    "homogenous": {'M': [5, 1/10000],
                   'D': [0.4, 1/10000],
                   'R': [0.3, 1/10000]}
}