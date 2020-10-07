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
    return round(agent.e * other.m, 2)

def temptation(agent, other):
    '''calculates how much temptation would change fitness'''
    return round((agent.e + agent.i * other.e) * other.m, 2)

def sucker(agent, other):
    '''calculates how much sucker would change fitness'''
    return 0

def punishment(agent, other):
    '''calculates how much punishment would chage fitness'''
    return round(agent.i * other.e * other.m, 2)

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