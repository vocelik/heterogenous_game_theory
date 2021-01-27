import math as m

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
<<<<<<< HEAD
    return round( (agent.d + agent.r) * other.m, 1)

def temptation(agent, other):
    '''calculates how much temptation would change fitness'''
    return round( (agent.d + 2 * agent.r) *  other.m, 1)
=======
    return round(agent.e * other.m, 2)

def temptation(agent, other):
    '''calculates how much temptation would change fitness'''
    return round((agent.e + agent.i * other.e) * other.m, 2)
>>>>>>> 2a27685c6f91b87ce259036064804a7d644831c2

def sucker(agent, other):
    '''calculates how much sucker would change fitness'''
    return 0

def punishment(agent, other):
    '''calculates how much punishment would chage fitness'''
<<<<<<< HEAD
    return round(agent.r * other.m, 1)
=======
    return round(agent.i * other.e * other.m, 2)
>>>>>>> 2a27685c6f91b87ce259036064804a7d644831c2

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