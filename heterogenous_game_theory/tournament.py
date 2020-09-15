import sys
import networkx as nx
import numpy as np
import time
from .strategies import cooperate, defect, tit_for_tat, generous_tit_for_tat, win_stay_lose_shift, APavlov, tit_for_two_tats
from .initialize_agents import *
from itertools import combinations, cycle
from .payoff_functions import default_payoff_functions, traditional_payoff_functions, selfreward
from .enums import to_outcome, Action, random_action

class Tournament:
    """
    TO DO
    """
    def __init__(
        self, 
        agents, 
        max_rounds, 
        strategy_list, 
        payoff_functions = default_payoff_functions, # rewards that agents get, defaults to the functions described in the paper.
        surveillance_penalty = True,
        penalty_dict = {cooperate: 1, defect: 1, tit_for_tat: 0.95, generous_tit_for_tat: 0.95, win_stay_lose_shift: 0.95, APavlov: 0.95, tit_for_two_tats: 0.95}, 
        noise = 0.1,
    ):
        """
        parameters:
            - agents: Agent
            - max_rounds: int
            - strategy_list: list
            - payoff_functions: dict
        """
        # meta data
        self.max_rounds: int = max_rounds
        self.strategy_list: list = strategy_list
        self.payoff_functions: dict = payoff_functions
        self.surveillance_penalty = surveillance_penalty
        self.penalty_dict = penalty_dict
        self.noise = noise
        
        # Data from the simulations will be stored in an NetworkX-graph
        self.graph = self.init_graph(agents, self.payoff_functions)
        
        # results of a the simulations
        self.fitness_results = np.zeros((len(self.agents()), max_rounds))
        self.evolution = [] # Todo: add evolution
        self.strategy_evolution = []
        self.winning_strategies = []
        # state variables
        self.round = 0
        self.is_done = False
       
    @staticmethod
    def init_graph(agents, payoff_functions):
        """
        initialize the graph (form the NetworkX library), that is used to store
        data from the simulation. Nodes in this graph store agents, and edges
        store the data associated with games between these agents (history, 
        payoff values, distance).
        
        parameters:
            - agents: list of agent, agents that take part in the Tournament
        """
        # we need to use DiGraph (directional edges) since Temptation and Sucker events are asymmetrical
        graph = nx.DiGraph() 
        
        # add nodes to graph
        for agent in agents:
            graph.add_node(agent)
            
        # add edges to graph
        # loop through all combinations of agents in the tournament.
        # the index of agent 1 (c1) is always less than that of agent 2 (c2)
        for c1, c2 in combinations(agents, 2):
            # calculate the payoff values that are associated with games 
            # between these two agents, and store them as tupples
            # R: reward, P: punishment, T: temptation, S: sucker
            RR = (payoff_functions['R'](c1,c2), 
                  payoff_functions['R'](c2,c1))
            PP = (payoff_functions['P'](c1,c2), 
                  payoff_functions['P'](c2,c1))
            TS = (payoff_functions['T'](c1,c2), 
                  payoff_functions['S'](c2,c1))
            ST = (payoff_functions['S'](c1,c2), 
                  payoff_functions['T'](c2,c1))
            
            # check if condition T > R > P > S is satisfied.
            assert RR[0] <= TS[0], f'reward was greater than temptation for agents {c1.name} and {c2.name}'
            assert RR[1] <= ST[1], f'reward was greater than temptation for agents {c1.name} and {c2.name}'
            assert ST[0] <= PP[0], f'sucker was greater than punishment for agents {c1.name} and {c2.name}'
            assert TS[1] <= PP[1], f'sucker was greater than punishment for agents {c1.name} and {c2.name}'
            
            # initialize all edges
            graph.add_edge(
                c1, 
                c2,
                # we add data to each edge
                history_1 = [], # list to accumulate tuples of actions
                history_2 = [],
                RR = RR, 
                PP = PP,
                TS = TS,
                ST = ST
            )
        
        return graph
    
    def agents(self):
        """
        return:
            - agents partaking in this tournament
        """
        return self.graph.nodes()
    
    # Todo: this now gives an assertion error
    def init_strategies(self, agents = None, strategy = None):
        """
        initizalize strategy for given agents. 
        
        parameters:
            agents: list or None, agents to change strategy, if None, then all agents are used
            strategy: list or None, stratey to be given to agents, if None, then echt agent
                      is randomly assigned a strategy from self.strategy_list.
        
        side effects:
            - changes the agents strategy
            
        example:
            >>> tournament = Tournament(...)
            >>> tournament.init_strategies(china, cooperate)
        """
        assert self.round == 0
        
        if isinstance(agents, Agent):
            agents = [agents]
        
        agents = agents or self.agents()
        
        for agent in agents:
            agent.change_strategy(
                    0, strategy or np.random.choice(self.strategy_list)
                    )
        
    def init_fitness(self, init_fitnes_as_m, agents = None):
        """
        initizalize the fiteness for each agent
        
        parameters:
            - init_fitnes_as_m: bool, if agents should start out with fitness
                                equal to their market size.
        
        side effects:
            - changes `fitness` and `fitness_history` for all agents
        """
        agents = agents or self.agents()
        
        for agent in agents:
            # todo: think if this logic has a better place in the agent class itself..
            agent.fitness = agent.m if init_fitnes_as_m else 0
            agent.fitness_history = [agent.fitness]
        
    def agents_per_strategy_dict(self):
        """
        return:
            - dictionary where the keys are strategies and the values the number of agents that play this strategy
            
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies(None, cooperate)
            >>> tournament.agents_per_strategy_dict()
            {cooperate: 100, defect: 0}
        """
        d = {strategy: 0 for strategy in self.strategy_list}
        for agent in self.agents():
            d[agent._strategy] += 1
        return d
        
    def agents_per_strategy_history(self):

        result = self.agents_per_strategy_dict()
        self.strategy_evolution.append(result)


    def one_strategy_left(self, strategy_n_agents_dict):
        """
        returns:
            - True if there is only one strategy left in the simulation
            
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies(None, cooperate)  # set al agents to cooperating strategy
            >>> tournament.one_strategy_left()
            True
        """
        for value in strategy_n_agents_dict.values():
            if value == len(self.agents()):
                return True
        return False

    
    def change_a_strategy(self, mutation_rate, round_num):
        """
        Change the strategy of a random agent, to become the strategy of a
        'winning agent', that was selected with probabilites proporitonal to 
        the fitness. This way strategies that do well in the tournament will
        spread through agents.
        
        Sometimes, in stead of the above, there will be a change in strategy
        that is entirely random.
        
        parameters:
            - mutation_rate: probabilitie of a random strategy change
            
        side effect:
            - changes a agents strategy
            
        """
        
        agent_list = list(self.agents())

        N = len(agent_list)
        
        # randomly select a agent that will lose its strategy to the winning strategy
        elimiation_idx = np.random.randint(N)
        losing_agent = agent_list[elimiation_idx]
        losing_strategy = losing_agent.get_current_strategy()
        
        # select a winning strategy
        mutation = bool(np.random.binomial(1, mutation_rate))
        if mutation:
            # in stead of changing a strategy according to the rules of the simmulation
            # we sometimes have a random mutation
            winning_strategy = np.random.choice(self.strategy_list)
            winning_agent = 'random_mutation'
        else:
            # we select a winning strategy with the probabilites of 'how much fitness each strategy has'
            fitness_scores = [agent.fitness for agent in agent_list]
            fitness_scores_non_neg = [max(0, fitness) for fitness in fitness_scores]
            total_fitness = sum(fitness_scores_non_neg)
            probabilities = [fitness_scores_non_neg[j]/total_fitness for j in range(N)] # errors if total fitness becomes negative...
            
            # select a random agent, with probabilities by normalized fitnesses
            reproduce_idx = np.random.choice(range(N), p=probabilities)
            winning_agent = agent_list[reproduce_idx]
            winning_strategy = winning_agent.get_current_strategy()
        
        # actually CHANGE the strategy
        losing_agent.change_strategy(round_num, winning_strategy)
        
        # for logging
        return losing_agent, winning_strategy 
           
    def check_all_strategies_initialized(self):
        """
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.check_all_strategies_initialized()
            False
            >>> tournament.init_strategies()
            >>> tournament.check_all_strategies_initialized()
            True
        """
        for agent in self.agents():
            if agent.get_current_strategy() is None:
                print(f'WARNING: {agent} has no initizalized strategy')

    def play_prisoners_dilema(self, agent_1, agent_2, game):
        """
        parameters:
            - coutry_1, agent_2: agent
            - game: dict, data associated with the game between agent_1 and agent_2 
            
        side effects:
            - appends history of game
            - changes fitness of agent_1 and agent_2
            
        example:
            >>> agent_1, agent_2, data = self.graph.get_edge(india, japan)
            >>> self.play_prisoners_dilema(agent_1, agent_2, data)
            XXX check this example
        """
            
        # let agents' strategies choose an action
        action_1 = agent_1.select_action(game['history_1'], game['history_2'], self.noise)
        action_2 = agent_2.select_action(game['history_2'], game['history_1'], self.noise)
        
        # append game history
        game['history_1'].append(action_1)
        game['history_2'].append(action_2)
        
        # get payoff values
        outcome = to_outcome(action_1, action_2)
        Δfitness_1,  Δfitness_2 = game[outcome] #self.graph.get_edge_data(agent_1, agent_2)[outcome]
 
        if self.surveillance_penalty:
            # to simmulate the effort that it takes to take a certain strategy
            # for some strategies a penalty is added, so that only part of the
            # change in fitness is really received.
            Δfitness_1 *= self.penalty_dict[agent_1.get_current_strategy()]
            Δfitness_2 *= self.penalty_dict[agent_2.get_current_strategy()]
            
        #change fitness
        agent_1.fitness += round(Δfitness_1)
        agent_2.fitness += round(Δfitness_2)
        
        return Δfitness_1, Δfitness_2, outcome
        
    def play(self, self_reward, playing_each_other, nr_strategy_changes, mutation_rate):

        """
        parameters:
            - self_reward: function or None, None indicates agents do not get 
                           reward from their internal market, if they do, self_reward
                           should be the function of the agent that gives them 
                           self_reward.
            - playing_each_other: bool, if agents play prisoners-dilema's with each
              other, and get/lose fitness from this
            - nr_strategy_changes: int, number of strategy-changes that occura
              after each round
              
        example:
            >>> tournament = Tournament(XXX)
            >>> tournament.init_strategies()
            >>> tournament.play(XXX)
            XXXXXXX
        """
        if self.is_done:
            print("WARNING: you are playing a tournament that has already been played. This will accumulate more"\
                  "data in the graph, which is probably incorrect. You probably want to re-initalize the tournament and"\
                  "agents, or refresh the kernel")
        
        strategies_initialized = self.check_strategies_initialized()

        if not strategies_initialized:
            print(f'All agents mus have initialized strategies, this can be done using the init_strategies method')
            return False


        # Start the tournament
        start_time = time.time()
        spinner = cycle(['-', '/', '|', '\\'])

        print("Tournament has started...")
        
        for i in range(self.max_rounds):
            
            self.round += 1
            sys.stdout.write(next(spinner))   # write the next character
            
            if self_reward:
                # agents get fitness from their own internal market
                for agent in self.agents():
                    agent.fitness += self_reward(agent)
            
            if playing_each_other: 
                # this can be switched to False to create a control-simulation for comparison
                for agent_1, agent_2, data in self.graph.edges(data=True): # data: include edge-attributes in the iteration
                    # todo: we are looping though edges twice.. this could be done only once.
                    self.play_prisoners_dilema(agent_1, agent_2, data)
                    
            for _ in range(nr_strategy_changes):
                # change {nr_strategy_changes} strategies
                losing_agent, winning_strategy = self.change_a_strategy(mutation_rate, self.round)
                self.winning_strategies.append(winning_strategy)
                
            # update fitness_histories
            for agent in self.agents():
                agent.fitness_history.append(agent.fitness)
                
            if self.one_strategy_left(self.agents_per_strategy_dict()):
                print(f'The process ended in {i+1} rounds\n Winning strategy: {list(self.agents())[0].get_current_strategy().name}')
                break

            self.agents_per_strategy_history()
            sys.stdout.flush()                # flush stdout buffer (actual character display)
            sys.stdout.write('\b')            # erase the last written char

        # flag that the tournament has ended
        self.is_done = True
        end_time = time.time() - start_time
        end_time = end_time // 60
        print("Tournament has ended. The simulation took " + str(int(end_time)) + " minutes.")
        
    def check_strategies_initialized(self):
        """
        check if all agents have a strategy.
        """
        for agent in self.agents():
            if agent.get_current_strategy() not in self.strategy_list:
                print(f'agent {agent.name} has strategy {agent.get_current_strategy()} which is not in the strategy_list')
                return False
        return True
            
    #Todo: remove initial fitness
    @classmethod
    def create_play_tournament(cls, 
                 agents, 
                 max_rounds, 
                 strategy_list=[defect, cooperate], 
                 payoff_functions=default_payoff_functions, 
                 surveillance_penalty = True,
                 self_reward = selfreward, #default function
                 playing_each_other=True,
                 nr_strategy_changes = 1,
                 mutation_rate =0.1,
                 init_fitnes_as_m=False,
                 noise = 0.1
                 ):
        """
        Create a tournament, initialize al the variables of the agents and
        then play the tournament.
        
        parameters:
            - agents: list, agents that take part in the tournament
            - max_rounds: int, maximum number of rounds, after which the tournament will stop
            - strategy_list: list, strategies that are played in the tournament
            - payoff_functions: functions to compute the changes in fitness, e.g. `default_payoff_functions` or `traditional_payoff_functions`
            - distance_function: function to rescale the distance. e.g. `lambda d:d` for linear scaleing and `lambda d: math.log(1+d)` for log-scaling
            - surveillance_penalty: bool, if agents should be penalized for playing certain strategies
            - self_reward: function or None, a function if agents should get reward from their internal market each round, 
                           otherwize, if agents should not get reward from their internal market, None
            - playing_each_other: bool, if agents should play prisoners delemma's with each other, set to false to create a control-group
            - nr_strategy_changes: int, number of strategy changes after eacht round
            - mutation_rate: probability that a strategy change is random
            - init_fitnes_as_m: bool, if agents start with self.fitness==self.m or self.fitness==0
        
        returns:
            the tournament object, with data from the simulation inside the
            graph attribute.
            
        example: 
            >>> tournament = Tournament.create_play_tournament(G50, 100, [cooperate, defect, tit_for_tat])
            ..... <code printed during the simulation>
            >>> tournament  # now this Tournament can be used for plotting and analytics
            <UvAAxelrod.tournament.Tournament at 0x7f115d97fcf8>
        """
        tournament = cls(agents, 
                 max_rounds, 
                 strategy_list, 
                 payoff_functions=payoff_functions, # rewards that agents get, defaults to the functions described in the paper.
                 surveillance_penalty = surveillance_penalty,
                 noise=noise
                 )
        tournament.init_strategies()
        tournament.init_fitness(init_fitnes_as_m=init_fitnes_as_m)
        
        tournament.play(self_reward, playing_each_other, nr_strategy_changes, mutation_rate)
        
        return tournament    
