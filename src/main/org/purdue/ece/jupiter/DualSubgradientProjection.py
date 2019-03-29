# This script simulates the projection sub-gradient algorithm in the dual for Exercise 3 of the ECE64700 HW3
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import numpy
from matplotlib import pyplot as plt


# This class encapsulates the solution to Part 4 of Exercise 3 in the ECE64700 HW3 employing the sub-gradient...
# ...projection algorithm in the dual
class DualSubgradientProjection(object):

    # Initialization sequence
    def __init__(self):
        print('[INFO] DualSubgradientProjection Initialization: Bringing things up...')
        # Number of receivers operating off of the central controller
        self.number_of_receivers = 5
        # /alpha_i
        self.alphas = [k + 30 for k in range(0, self.number_of_receivers)]
        # /beta_i
        self.betas = [k + 50 for k in range(0, self.number_of_receivers)]
        # Initial value of /lambda
        self.initial_lambda = 0.1
        # Default step size
        self.default_step_size = 1
        # Maximum Power = 100 dB
        self.max_power = 100
        # Maximum Bandwidth = 10 MHz
        self.max_bandwidth = 10
        # The iterations array modelling the x_axis for the convergence plot
        self.iterations = []
        # The lambda values array modelling the y_axis for the convergence plot
        self.lambdas = []
        # Infinity behavior modelling
        self.infinity = 10 ** 10
        # Confidence Bound for convergence modelling
        self.confidence_bound = 10

    # Projection Utility method
    def projection(self, value, lower_bound, upper_bound):
        if upper_bound == self.infinity:
            return max(lower_bound, value)
        else:
            if value < lower_bound:
                return lower_bound
            elif value > upper_bound:
                return upper_bound
        return value

    # Convergence check
    def has_it_converged(self, previous_value, current_value):
        if previous_value is not None:
            if previous_value == current_value and self.projection(current_value, 0, self.infinity) == current_value:
                return True
        return False

    # Get the sub-gradient
    def get_subgradient(self, previous_value):
        f_max = 0
        f_argmax = None
        bandwidth_allocation = [k - k for k in range(0, self.number_of_receivers)]
        for receiver in range(0, self.number_of_receivers):
            if (self.alphas[receiver] * self.betas[receiver]) > previous_value:
                try:
                    f = (self.alphas[receiver] * numpy.log(
                        (self.alphas[receiver] * self.betas[receiver]) / previous_value)) + (
                                previous_value / self.betas[receiver]) - self.alphas[receiver]
                except ZeroDivisionError as e:
                    print('[ERROR] DualSubgradientProjection get_subgradient: Division by zero exception - beta= {}, '
                          'lambda_k= {}, error_message= {}'.format(self.betas[receiver], previous_value, e))
                    return 0
                if f > f_max:
                    f_max = f
                    f_argmax = receiver
        bandwidth_allocation[f_argmax] = self.max_bandwidth
        power_allocation = [k - k for k in range(0, self.number_of_receivers)]
        for receiver in range(0, self.number_of_receivers):
            power_allocation[receiver] = (bandwidth_allocation[receiver] / self.betas[receiver]) * self.projection(
                ((self.alphas[receiver] * self.betas[receiver]) / previous_value) - 1, 0, self.max_power)
        return sum(power_allocation) - self.max_power

    # The optimization procedure
    def optimize(self):
        previous_value = None
        iteration_count = 0
        current_value = self.initial_lambda
        confidence = 0
        while confidence < self.confidence_bound and self.has_it_converged(previous_value, current_value) is False:
            if self.has_it_converged(previous_value, current_value):
                confidence += 1
            self.lambdas.append(current_value)
            iteration_count += 1
            self.iterations.append(iteration_count)
            previous_value = current_value
            current_value = self.projection(
                previous_value + ((1 / iteration_count) * self.get_subgradient(previous_value)), 0,
                self.infinity)
        print('[INFO] DualSubgradientProjection optimize: The Optimization sequence for lambda has been completed...')

    # Visualize the convergence or divergence of the algorithm with respect to varying step sizes
    def visualize(self):
        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.lambdas, linewidth=1.0, marker='o', color='r')
        fig.suptitle('Convergence Visualization of the Projection Subgradient Algorithm in the Dual', fontsize=12)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.set_ylabel('Lambda', fontsize=14)
        plt.show()

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] DualSubgradientProjection Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] DualSubgradientProjection main: Staring system simulation...')
    dualSubgradientProjection = DualSubgradientProjection()
    dualSubgradientProjection.optimize()
    dualSubgradientProjection.visualize()
    print('[INFO] DualSubgradientProjection main: System simulation has been stopped...')
