# Importing liabraries
import numpy as np
import random
from itertools import groupby
from itertools import product


# Creating environment class
class TicTacToe():

    def __init__(self):                       
        """ initializing the Tic-Tac-Toe board """
        
        # initialising the state as an array
        self.state = [np.nan for _ in range(9)]  
        
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] 

        
    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: 
        Input state = [1, 2, 5, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        if ((sum(curr_state[0:3:1]) == 15) or     #row1
            (sum(curr_state[3:6:1]) == 15) or    #row2
            (sum(curr_state[6:9:1]) == 15) or    #row3
            (sum(curr_state[0:9:3]) == 15) or    #column1
            (sum(curr_state[1:9:3]) == 15) or    #column2
            (sum(curr_state[2:9:3]) == 15) or    #column3
            (sum(curr_state[0:9:4]) == 15) or    #diagonal1
            (sum(curr_state[2:7:2]) == 15)):      #diagonal2
            return True
        else:
            return False
            

    def is_terminal(self, curr_state):              # Returns terminal state of an episode.
        """ Terminal state could be winning state or when the board is filled up"""

        if self.is_winning(curr_state) == True:    #Returns true,'win' if episode is terminated by winning.
            return True, 'Win'
        elif len(self.allowed_positions(curr_state)) ==0:  #Returns true,'Tie' if episode is finishd by draw.
            return True, 'Tie'
        else:                                       #Returns False,'Resume' if episode is not terminated.
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board. Agent uses odd numbers(1,3,5,7,9) and environment uses even numbers(2,4,6,8)"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]
        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: 
        Input state = [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. 
        Example: 
        Input state = [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        next_state = self.state_transition(curr_state, curr_action)
        terminal , current_condition = self.is_terminal(next_state)
        if current_condition ==  'Win':
            reward = 10
            info = 'agent won'
        elif current_condition == 'Tie':
            reward = 0
            info = 'draw'
        else:
            env_act = random.choice(list(self.action_space(curr_state)[1]))
            next_state = self.state_transition(next_state,env_act)
            terminal , current_condition = self.is_terminal(next_state)
            if current_condition ==  'Win':
                reward = -10
                info = 'env won'
            elif current_condition == 'Tie':
                reward = 0
                info = 'draw'
            else :
                reward = -1
                info = 'continue'
        return reward , next_state, terminal, info
    
