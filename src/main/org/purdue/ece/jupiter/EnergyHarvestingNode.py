# This entity solves for the optimal policy in a energy harvesting radio node using concepts from Markov Decision...
# ...Processes. Using this optimal policy, we further evaluate the system performance and present numerical results.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import sys
import traceback
import itertools
from matplotlib import pyplot as plt


# This class encapsulates the decision engine of a energy harvesting radio node with inputs from...
# ...the Channel subsystem for the fading coefficient realizations and...
# ...the Energy Harvesting subsystem for the amount of ambient energy harvested
class EnergyHarvestingNode(object):
    # The number of realizations of the fading channel coefficients
    NUMBER_OF_FADING_STATE_REALIZATIONS = 10

    # The maximum allowed battery energy multiplier which implies that the maximum allowed energy in the battery is ME_0
    MAX_BATTERY_ENERGY_MULTIPLIER = 10

    # The probability of harvesting E_0 units of energy in a time slot by the energy harvesting subsystem
    ENERGY_HARVESTING_SUCCESS_PROBABILITY = 0.1

    # The discount factor for this INFINITE HORIZON DISCOUNTED REWARD PROBLEM FORMULATION
    DISCOUNT_FACTOR = 0.9

    # The smallest non-zero battery energy magnitude, i.e. E_0 units
    MIN_BATTERY_ENERGY = 1

    # State String Separator
    STATE_STRING_SEPARATOR = '_'

    # Confidence Bound
    CONFIDENCE_BOUND = 30

    # Get all the possible states - A state is considered to be a 2-tuple or a pair <Q_k, g_k>
    def get_all_possible_states(self):
        # m \in \{0, 1, 2, ..., M\}
        all_possible_battery_status_realizations = [k * self.MIN_BATTERY_ENERGY
                                                    for k in range(0, self.MAX_BATTERY_ENERGY_MULTIPLIER + 1)]
        # i \in \{1, 2, 3, ..., N\}
        all_possible_fading_state_realizations = [i for i in range(1, self.NUMBER_OF_FADING_STATE_REALIZATIONS + 1)]
        # Return an array of <Q_k, g_k> 2-tuples
        return list(itertools.product(all_possible_battery_status_realizations, all_possible_fading_state_realizations))

    # Initialization sequence
    def __init__(self):
        print('[INFO] EnergyHarvestingNode DecisionEngine-Initialization: Bringing things up...')
        # Assimilate all the possible states for the system
        self.all_possible_states = self.get_all_possible_states()
        # Initial arbitrary policy
        self.initial_policy = {str(s[0]) + self.STATE_STRING_SEPARATOR + str(s[1]): 0
                               for s in self.all_possible_states}
        # The optimal policy returned by value iteration
        self.value_iteration_optimality = dict()
        # The optimal policy returned by policy iteration
        self.policy_iteration_optimality = dict()
        # Value Iteration Value function array - Y-axis
        self.value_iteration_value_function_array = list()
        # Value Iteration Iterations array - X-axis
        self.value_iteration_iterations_array = list()
        # Policy Iteration Value function array - Y-axis
        self.policy_iteration_value_function_array = list()
        # Policy Iteration Iterations array - X-axis
        self.policy_iteration_iterations_array = list()

    # Convergence check for value iteration
    @staticmethod
    def has_value_iteration_converged(previous_value_function, current_value_function):
        if len(current_value_function.keys()) is not 0:
            for state, value in previous_value_function.items():
                if previous_value_function[state] != current_value_function[state]:
                    return False
            return True
        return False

    # Convergence check for policy iteration
    @staticmethod
    def has_policy_iteration_converged(previous_policy, current_policy):
        if len(current_policy.keys()) is not 0:
            for state, action in previous_policy.items():
                if previous_policy[state] != current_policy[state]:
                    return False
            return True
        return False

    # Get all the available actions in that state
    def get_available_actions(self, state):
        battery_state, fading_state = state.split(self.STATE_STRING_SEPARATOR)
        # Battery's dead - you can't do anything
        if int(battery_state) == 0:
            # Come up with a different logic that doesn't involve unnecessary list creations and...
            # ...unnecessary if_else checks in the calling routine.
            # NOTE: 0 corresponds to the node being idle
            return [0]
        return [k for k in range(0, int(fading_state) + 1)]

    # Get the transition probability based on the current_state, the next_state, and the action, i.e. P(s'|s,u)
    def get_transition_probability(self, current_state, next_state, action):
        # I don't worry about the fading state because the transitions are i.i.d with P(g_{k+1}|g_k) = P(g_{k+1}) = 1/N
        # I need to worry about the battery state
        # Remember the state string structure Q_k SEPARATOR g_k
        current_battery_state = int(current_state.split(self.STATE_STRING_SEPARATOR)[0])
        projected_next_battery_state = int(next_state.split(self.STATE_STRING_SEPARATOR)[0])
        if action == 0 and projected_next_battery_state == current_battery_state:
            # (1-p)/N: unsuccessful harvesting and idle node
            return (1 - self.ENERGY_HARVESTING_SUCCESS_PROBABILITY) / self.NUMBER_OF_FADING_STATE_REALIZATIONS
        elif action == 0 and projected_next_battery_state == current_battery_state + 1:
            # p/N: successful harvesting and idle node
            return self.ENERGY_HARVESTING_SUCCESS_PROBABILITY / self.NUMBER_OF_FADING_STATE_REALIZATIONS
        elif action > 0 and projected_next_battery_state == current_battery_state:
            # p/N: successful harvesting and active node (energy consumed and harvested)
            return self.ENERGY_HARVESTING_SUCCESS_PROBABILITY / self.NUMBER_OF_FADING_STATE_REALIZATIONS
        elif action > 0 and projected_next_battery_state == current_battery_state - 1:
            # (1-p)/N: unsuccessful harvesting and active node (energy consumed but not harvested)
            return (1 - self.ENERGY_HARVESTING_SUCCESS_PROBABILITY) / self.NUMBER_OF_FADING_STATE_REALIZATIONS
        else:
            # The rest can't happen because of our system model
            return 0

    # Get the optimal policy from value iteration
    def get_value_iteration_optimality(self):
        # A string representation of the states
        all_possible_states = [str(s[0]) + self.STATE_STRING_SEPARATOR + str(s[1]) for s in self.all_possible_states]
        # The current value function
        current_value_function = dict()
        # The initial value function as a dict referred to by the string representation of the process states
        initial_value_function = {s: 0.00001 for s in all_possible_states}
        self.value_iteration_iterations_array.append(0)
        self.value_iteration_value_function_array.append(initial_value_function)
        # The previous value function
        previous_value_function = initial_value_function
        # The optimal policy collection for updates during the iterative process
        optimal_policy = dict()
        # Iteration count
        iteration_count = 0
        # Confidence counter
        confidence = 0
        # Convergence check
        while (confidence >= self.CONFIDENCE_BOUND and
               self.has_value_iteration_converged(previous_value_function, current_value_function)) is False:
            if self.has_value_iteration_converged(previous_value_function, current_value_function):
                confidence += 1
            value_function_dict = dict()
            previous_value_function = (lambda: previous_value_function, lambda: current_value_function)[len(
                current_value_function.keys()) is not 0]()
            for current_state in all_possible_states:
                # Get all the actions available to the MDP agent in this state
                available_actions = self.get_available_actions(current_state)
                # Maximum value initialization for the maximization process
                maximum_value = 0
                # Maximizing action
                maximizing_action = None
                for action in available_actions:
                    internal_sum = 0
                    for next_state in all_possible_states:
                        internal_sum += self.get_transition_probability(current_state, next_state, action) * \
                                        previous_value_function[next_state]
                    # Evaluate the reward corresponding to the action
                    reward = action
                    value = reward + (self.DISCOUNT_FACTOR * internal_sum)
                    if value > maximum_value:
                        maximum_value = value
                        maximizing_action = action
                value_function_dict[current_state] = maximum_value
                current_value_function[current_state] = maximum_value
                optimal_policy[current_state] = maximizing_action
            self.value_iteration_value_function_array.append(value_function_dict)
            iteration_count += 1
            self.value_iteration_iterations_array.append(iteration_count)
        self.value_iteration_optimality = optimal_policy

    # Get the optimal policy from policy iteration
    def get_policy_iteration_optimality(self):
        # A string representation of the states
        all_possible_states = [str(s[0]) + self.STATE_STRING_SEPARATOR + str(s[1]) for s in self.all_possible_states]
        # Current value function
        current_value_function = dict()
        # The initial value function as a dict referred to by the string representation of the process states
        initial_value_function = {s: 0.00001 for s in all_possible_states}
        self.policy_iteration_iterations_array.append(0)
        self.policy_iteration_value_function_array.append(initial_value_function)
        # The previous value function
        previous_value_function = initial_value_function
        # Previous policy
        previous_policy = self.initial_policy
        # Current policy
        current_policy = dict()
        # Iteration count
        iteration_count = 0
        # Confidence counter
        confidence = 0
        while (confidence >= self.CONFIDENCE_BOUND and
               self.has_policy_iteration_converged(previous_policy, current_policy)) is False:
            if self.has_policy_iteration_converged(previous_policy, current_policy):
                confidence += 1
            previous_policy = (lambda: previous_policy, lambda: current_policy)[len(current_policy.keys()) is not 0]()
            value_function_dict = dict()
            previous_value_function = (lambda: previous_value_function, lambda: current_value_function)[len(
                current_value_function.keys()) is not 0]()
            # Policy Evaluation
            for current_state in all_possible_states:
                action = previous_policy[current_state]
                # Evaluate the reward corresponding to the action
                reward = action
                internal_sum = 0
                for next_state in all_possible_states:
                    internal_sum += self.get_transition_probability(current_state, next_state, action) * \
                                    previous_value_function[next_state]
                current_value_function[current_state] = reward + (self.DISCOUNT_FACTOR * internal_sum)
            # Policy Improvement
            for _current_state in all_possible_states:
                # Get all the actions available to the MDP agent in this state
                available_actions = self.get_available_actions(_current_state)
                # Maximum value during the maximization process
                maximum_value = 0
                # Maximizing action during the maximization process
                maximizing_action = None
                for _action in available_actions:
                    # Evaluate the reward corresponding to the action
                    _reward = _action
                    _internal_sum = 0
                    for _next_state in all_possible_states:
                        _internal_sum += self.get_transition_probability(_current_state, _next_state, _action) * \
                                         current_value_function[_next_state]
                    value = _reward + (self.DISCOUNT_FACTOR * _internal_sum)
                    if value > maximum_value:
                        maximum_value = value
                        maximizing_action = _action
                current_policy[_current_state] = maximizing_action
                value_function_dict[_current_state] = maximum_value
            self.policy_iteration_value_function_array.append(value_function_dict)
            iteration_count += 1
            self.policy_iteration_iterations_array.append(iteration_count)
        self.policy_iteration_optimality = current_policy

    # Evaluate the performance of the optimal policy vs the battery status and the fading state coefficients
    def evaluate_performance_of_optimality(self):
        # A random state choice for evaluation
        state_choice = '9_10'
        value_iteration_modified_value_iterations_array = [value[state_choice]
                                                           for value in self.value_iteration_value_function_array]
        policy_iteration_modified_value_iterations_array = [value[state_choice]
                                                            for value in self.policy_iteration_value_function_array]
        # Plot value iteration and policy iteration optimalities as a function of the iteration index
        fig, ax = plt.subplots()
        ax.plot(self.value_iteration_iterations_array, value_iteration_modified_value_iterations_array, linewidth=1.0,
                marker='o', color='r', label='Value Iteration')
        ax.plot(self.policy_iteration_iterations_array, policy_iteration_modified_value_iterations_array, linewidth=1.0,
                marker='o', color='b', label='Policy Iteration')
        fig.suptitle('Convergence Visualization of the Value Iteration and Policy Iteration Algorithms for the '
                     'Decision Engine in the given Energy Harvesting System considering system state '
                     '$(Q_k=9,\ g_k=\gamma_{10}$)', fontsize=10)
        ax.set_xlabel('Number of Iterations', fontsize=14)
        ax.set_ylabel('$V(Q_k=9,\ g_k=\gamma_{10}$)', fontsize=14)
        ax.legend()
        plt.show()

    # Plot the optimal policy vs battery state and fading state; to do so, plot N curves, where the ith one is...
    # ...the optimal policy vs battery state under fading state \gamma_i
    def evaluate_performance_globally(self):
        # The figure
        fig, ax = plt.subplots()
        # Colors tuple
        colors = ['b', 'r', 'y', 'k', 'g', 'c', 'm', (0.6, 0.7, 0.9), (0.3, 0.8, 0.8), (0.8, 0.1, 0.8)]
        # The external control variable
        for fading_state in range(1, self.NUMBER_OF_FADING_STATE_REALIZATIONS + 1):
            # The X-Axis
            battery_states = list()
            # The Y-Axis
            optimal_actions = list()
            for battery_state in range(0, self.MAX_BATTERY_ENERGY_MULTIPLIER + 1):
                battery_states.append(battery_state)
                system_state = str(battery_state) + self.STATE_STRING_SEPARATOR + str(fading_state)
                optimal_action = self.policy_iteration_optimality[system_state]
                optimal_actions.append(optimal_action)
            ax.plot(battery_states, optimal_actions, linewidth=1.0, marker='o', color=colors[fading_state - 1],
                    label='Fading State = $\gamma_{' + str(fading_state) + '}$')
        fig.suptitle('Optimal policy versus Battery State and Fading State for the Decision Engine in the given '
                     'Energy Harvesting System', fontsize=10)
        ax.set_xlabel('Battery State $Q_k$', fontsize=14)
        ax.set_ylabel('Optimal Action $u_k$ (number of bits to transmit)', fontsize=14)
        ax.legend()
        plt.show()

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] EnergyHarvestingNode DecisionEngine-Termination: Tearing things down...')
        self.all_possible_states = None


# Run Trigger
# This call simulates the call to the DecisionEngine from the parent class in the complete design of the...
# ...Energy Harvesting Radio Node
if __name__ == '__main__':
    print('[INFO] EnergyHarvestingNode main: Starting system simulation...!')
    print('[INFO] EnergyHarvestingNode main: Creating DecisionEngine instance...!')
    try:
        # Create the DecisionEngine instance
        decisionEngine = EnergyHarvestingNode()
        # Get the optimal policy from value iteration
        decisionEngine.get_value_iteration_optimality()
        # Get the optimal policy from policy iteration
        decisionEngine.get_policy_iteration_optimality()
        # Evaluate the performance of the optimal policy vs the battery status and the fading state coefficients
        decisionEngine.evaluate_performance_of_optimality()
        # Evaluation of the optimal policy versus system state
        decisionEngine.evaluate_performance_globally()
        print('[INFO] EnergyHarvestingNode main: Found and evaluated the optimal policy! '
              'Shutting the decision engine down...!')
    except Exception as e:
        print('[ERROR] EnergyHarvestingNode main: An exception was caught while solving for and evaluating the '
              'optimal policy! - {}'.format(e))
        traceback.print_exc()
        sys.exit(1)
    print('[INFO] EnergyHarvestingNode main: System simulation ended...!')
