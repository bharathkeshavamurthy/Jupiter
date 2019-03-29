# This Python script simulates Optimal Power and Bandwidth Allocation strategies for receivers in a centralized...
# ...setting comprising Gaussian broadcast channels.
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='bkeshava', api_key='RHqYrDdThygiJEPiEW5S')


# This class encapsulates the optimization solution arrived at using the Sub-Gradient Projection Algorithm in the Dual
class OptimalPowerAndBandwidthAllocation(object):

    # The Initialization sequence
    def __init__(self):
        print('[INFO] OptimalPowerAndBandwidthAllocation Initialization: Bringing things up...')
        # The number of receivers in the radio environment
        self.number_of_receivers = 5
        # A collection of \alpha_is
        self.alphas = [((k + 10) / self.number_of_receivers) for k in range(0, self.number_of_receivers)]
        # A collection of \beta_is
        self.betas = [((k + 10) / self.number_of_receivers) for k in range(0, self.number_of_receivers)]
        # Maximum allowed power
        self.max_power = 0.5
        # Maximum allowed bandwidth
        self.max_bandwidth = 1.0
        # Initial Value of \lambda
        self.initial_lambda = 1
        # The confidence bound for convergence analysis
        self.confidence_bound = 10
        # The X-Axis for visualizing the convergence
        self.x_axis = [0]
        # The Y-Axis for visualizing the convergence
        self.y_axis = [self.initial_lambda]
        # Maximum number of iterations for divergence analysis
        self.max_iterations = 100000

    # A Simple Generalized Projection utility method
    @staticmethod
    def projection(value, lower_bound, upper_bound):
        # The upper bound is infinity
        if upper_bound is None:
            return max(lower_bound, value)
        # A well-defined upper bound exists
        if lower_bound <= value <= upper_bound:
            return value
        return (lambda: lower_bound, lambda: upper_bound)[value > upper_bound]()

    # Get the subgradient from the P_i and W_i values of the current iteration
    def get_subgradient(self, lambda_value):
        valid_subset = []
        # Find the reduced subset of valid receivers with {\alpha_i * \beta_i} > {\lambda_k * ln(2)}
        for receiver in range(0, self.number_of_receivers):
            if (self.alphas[receiver] * self.betas[receiver]) > (lambda_value * np.log(2)):
                valid_subset.append(receiver)
        max_value = -10 ** 100
        argmax = None
        if len(valid_subset) is 0:
            print('[ERROR] OptimalPowerAndBandwidthAllocation get_subgradient: Valid receiver subset is empty...')
            return -1 * self.max_power
        # argmax operation
        for valid_receiver in valid_subset:
            f = (self.alphas[valid_receiver] * (
                np.log((self.alphas[valid_receiver] * self.betas[valid_receiver]) / (lambda_value * np.log(2))))) + (
                        lambda_value / self.betas[valid_receiver]) - (self.alphas[valid_receiver] / np.log(2))
            if f > max_value:
                max_value = f
                argmax = valid_receiver
        # Calculate and return the subgradient
        return ((self.max_bandwidth / self.betas[argmax]) *
                (((self.alphas[argmax] * self.betas[argmax]) / (lambda_value * np.log(2))) - 1)) - self.max_power

    # Convergence check routine
    def convergence(self, previous_lambda, current_lambda):
        # Check for consistency and for the stopping condition [\lambda_k]^+ = \lambda_k
        if previous_lambda == current_lambda and self.projection(current_lambda, 0, None) == current_lambda:
            return True
        return False

    # The Sub-Gradient Projection Algorithm in the Dual
    def optimize(self, _step_size):
        confidence = 0
        previous_lambda = 0
        current_lambda = self.initial_lambda
        k = 1
        while k < self.max_iterations and (
                confidence < self.confidence_bound and (self.convergence(previous_lambda, current_lambda) is False)):
            if self.convergence(previous_lambda, current_lambda):
                confidence += 1
            previous_lambda = current_lambda
            current_lambda = self.projection((previous_lambda +
                                              ((lambda: _step_size, lambda: (1 / k))[_step_size is None]() *
                                               self.get_subgradient(previous_lambda))), 0, None)
            self.x_axis.append(k)
            self.y_axis.append(current_lambda)
            k += 1
        print('[INFO] OptimalPowerAndBandwidthAllocation optimize: The optimization sequence has been completed...')

    # The Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] OptimalPowerAndBandwidthAllocation Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] OptimalPowerAndBandwidthAllocation main: Starting system simulation')
    # Create the allocator instance
    optimalPowerAndBandwidthAllocator = OptimalPowerAndBandwidthAllocation()
    # Run the Sub-Gradient Projection Algorithm in the Dual
    optimalPowerAndBandwidthAllocator.optimize(None)
    # Visualize the results post-convergence
    # Data Trace
    data_trace = go.Scatter(
        x=optimalPowerAndBandwidthAllocator.x_axis,
        y=optimalPowerAndBandwidthAllocator.y_axis,
    )
    # Plot Layout
    layout = dict(
        title='Convergence Visualization of the dual variables during the course of the Sub-Gradient Projection '
              'Algorithm', xaxis=dict(title='Iterations'), yaxis=dict(title='$\lambda$'))
    fig = dict(data=[data_trace], layout=layout)
    # Online plot using bkeshava's plotly service
    py.iplot(fig, filename='Optimal_Power_And_Bandwidth_Allocation_Convergence_Analysis')
    # Visualization completed
    print('[INFO] OptimalPowerAndBandwidthAllocation main: System simulation completed...')
