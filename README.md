# Heterogenous Game Theory

__NOTE__: if you are viewing this page in Google Chrome, we strongly advise to install the [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related).

## Introduction

This package is based on the [UvAAxelrod](https://github.com/RickGroeneweg/UvAAxelrod) package, which Rick Groeneweg, Vaclav Ocelik and Sebastian Krapohl developed for the research project GLOBAL CODE. The package simulates prisoner's dilemma between any number of _agents_ in a particular environment. Agents can, but need not, differ from one another based on three variables: _M_, _e_, and _i_. 

For any agent pair _x_,_y_ the payoffs are calculated as follows:

$$ R  = (x_{e} - x_{i} * y_{m}) * y_m$$
$$ T  = x_{e} * y_{m}$$
$$ S  = -x_{i} * y_{e} * y_{m}$$
$$ P  = 0$$

For all but the most extreme values of _M_, _e_, and _i_. This satisfies the condition for the prisoner's dilemma:

$$ T>R>P>S$$

The variables can be a constant, or can be drawn from a normal or pareto distribution. The user can specify any combination of constants and distribution. For example, _M_ can be set to 1000 with 0 variation, _i_ can have an average of _0.1_ and a standard deviation of _0.2_, and _e_ can have an average of _0.1_ and a shape of _4_. This allows the user to create a population of agents that is completely homogenous, completely heterogenous, and anywhere in between. 

## Instructions

Running a tournament is fairly straightforward. We import all files in the package, as wel as `numpy` and `scipy`. 

    import sys
    sys.path.insert(1, '../')
    from heterogenous_game_theory import *
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy as sp
    seed = 1024
    np.random.seed(seed)

Next, we create a set of agents with a particular distribution and check the parameters of these agents. Notice how we input "power" into `E[0]` to indicate we want a pareto distribution. If we set `homogenous = True`, we need not create `M`, `E`, or `I` separately.

    M = [1000, 2000]
    E = ["power", 0.1, 6]
    I = [0.1, 1/10000]

    agents = get_agents(homogenous = False, number_of_agents = 100, M = M, E = E, I = I)
    check_parameters(agents)
    compare_payoff_function(agents, default_payoff_functions)

`check_parameters` prints the characteristics of the first twenty agents, as well as a histogram of all agents' characteristics. `compare_payoff_function` prints the payoffs of the first ten agent pairs. These can be used in conjunction to quickly inspect if everything worked as intended.  

Finally, we run the tournament and save the results.

    tour = Tournament.create_play_tournament(
                    agents = agents, 
                    max_rounds = 10000, 
                    strategy_list = [defect, tit_for_tat, generous_tit_for_tat, cooperate], 
                    payoff_functions = default_payoff_functions, 
                    surveillance_penalty = True,
                    self_reward = selfreward, #default function
                    playing_each_other = True,
                    nr_strategy_changes = 10,
                    mutation_rate = 0.1,
                    init_fitnes_as_m = False,
                    noise = 0.05,
                    )
    draw_stack(tour)
    C_D_ratios_per_round_var(tour, constant = 1)
    outliers = count_outliers(tour, constants = np.arange(0.5, 3.1, 0.1))
    data = {'S.D.': list(outliers.keys()), 'Counts': list(outliers.values())}
    df = pd.DataFrame.from_dict(data)
    print(df)
    df.to_csv("Data/data_" + str(seed) + "_outliercounts.csv", encoding='utf-8', index = False, float_format='%.1f')
    save_tournament_csv(tour, type_of_tournament= "complete_heterogeneity", seed = str(seed))

`draw_stack` draws a stackplot of market share on the y-axis and round number on the x-axis. This plot allows us to inspect how the different strategies in our population have captured market share in the simulation. 

