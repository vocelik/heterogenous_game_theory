# Heterogenous Game Theory

__NOTE__: if you are viewing this page in Google Chrome, we strongly advise to install the [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/related).

## Introduction

This package is based on the [UvAAxelrod](https://github.com/RickGroeneweg/UvAAxelrod) package, which Rick Groeneweg, Václav Ocelík and Sebastian Krapohl developed for the research project GLOBAL CODE. The package simulates prisoner's dilemma between any number of _agents_ in a particular environment. Agents can, but need not, differ from one another based on three variables: _M_, _d_, and _r_. 

For any agent pair _x_,_y_ the payoffs are calculated as follows:

$$ T  = (x_{d} + 2 * x_{r}) * y_{m} $$
$$ R  = (x_{d} + x_{r}) * y_m $$
$$ P  = x_{r} * y_{m} $$
$$ S  = 0 $$

For all values of _M_, _d_, and _r_. This satisfies the condition for the prisoner's dilemma:

$$ T>R>P>S $$

and

$$ 2R > T + S $$

The variables can be a constant, or can be drawn from a normal or power distribution. The user can specify any combination of constants and distribution. For example, _M_ can be set to 5 with 0 variation, _d_ can have an average of _0.1_ and a standard deviation of _0.2_, and _r_ can have an average of _0.1_ and a shape of _4_. This allows the user to create a population of agents that is completely homogenous, completely heterogenous, and anywhere in between. 

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
    D = ["power", 0.1, 6]
    R = [0.1, 1/10000]

    agents = get_agents(homogenous = False, number_of_agents = 100, M = M, D = D, R = R)
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

![draw_stack](https://github.com/vocelik/heterogenous_game_theory/blob/master/images/draw_stack.png)

`C_D_ratios_per_round_var` plots the average cooperation ratio on the y-axis and the round number on the x-axis. The black line constitutes the average of the entire tournament. The yellow and red lines capture the upper and lower bounds of the desired standard deviation, the default being 1. Points outside the yellow and red lines signal periods of high or low levels of cooperation. This plot allows us to inspect how the average cooperation ratio evolves over the tournament, as well as how stable this level is. 

![coop_ratio](https://github.com/vocelik/heterogenous_game_theory/blob/master/images/cd_ratios.png)

The output of the simulation is saved in two csv files. The first saves the average cooperation rate of each round, while the second saves the absolute number of extreme periods of cooperation/defection for a standard deviation ranging from 0.5 to 3. A period starts whenever the line exits the area between the yellow and red line and ends whenever the line returns. 