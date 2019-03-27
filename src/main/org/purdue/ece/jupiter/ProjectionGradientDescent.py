# This script simulates the solution to the minimization problem outlined in the ECE64700 Homework Assignment III by...
# ... employing the Projection Gradient Descent Algorithm
# Author: Bharath Keshavamurthy
# Organization: School of Electrical & Computer Engineering, Purdue University
# Copyright (c) 2019. All Rights Reserved.

import math
from matplotlib import pyplot as plt


# This class encapsulates the simulation of the solution to the convex optimization problem laid down in Exercise 2...
# ... of the ECE64700 Homework Assignment III using the Projection Gradient Descent Algorithm
class ProjectionGradientDescent(object):

    # Initialization sequence
    def __init__(self):
        print('[INFO] ProjectionGradientDescent Initialization: Bringing things up...')
        # Initial point x_0\ =\ (0,\ 1) \in \mathcal{F}
        self.x_0 = (0, 1)
        # Default step size
        self.default_step_size = 0.1
        # The Line Segments
        # Use (0, 1000) to model (0, \infty) and Use (1000, 0) to model (\infty, 0)
        self.hyperplanes = [[(0, 1000), (0, 1)], [(0, 1), (0.4, 0.2)], [(0.4, 0.2), (1, 0)], [(1, 0), (1000, 0)]]
        # Iterations array which models the x-axis of the convergence plot
        self.iterations = []
        # Function values array which models the y-axis of the convergence plot
        self.function_values = []
        # Confidence Bound for Convergence
        self.confidence_bound = 10
        # Max Iterations allowed to account for divergence when using large step sizes
        self.max_iterations = 100000

    # Vector Projection Technique
    # https://en.wikipedia.org/w/index.php?title=Vector_projection&oldid=861961162#Vector_projection_2
    @staticmethod
    def vector_projection(point, hyperplane):
        x1, y1 = hyperplane[0]
        x2, y2 = hyperplane[1]
        x, y = point
        projection_go_ahead = False
        # TODO: Change this to simpler existence check in the polyhedron
        if (x < 0 or y < 0) or (((2 * x) + y) < 1) or ((x + (3 * y)) < 1):
            projection_go_ahead = True
        if projection_go_ahead:
            dot_product = ((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1))
            norm_square = ((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1))
            if norm_square == 0:
                print(
                    '[ERROR] ProjectionGradientDescent vector_projection: Error while evaluating the component of '
                    '\vec{a} along \vec{b} using vector projection techniques...')
                return None
            component_value = dot_product / norm_square
            if component_value < 0:
                return math.sqrt(((x - x1) * (x - x1)) + ((y - y1) * (y - y1))), (x1, y1)
            elif component_value > 1:
                return math.sqrt(((x - x2) * (x - x2)) + ((y - y2) * (y - y2))), (x2, y2)
            x_closest_point = x1 + (component_value * (x2 - x1))
            y_closest_point = y1 + (component_value * (y2 - y1))
            return (math.sqrt(
                ((x - x_closest_point) * (x - x_closest_point)) + ((y - y_closest_point) * (y - y_closest_point))),
                    (x_closest_point, y_closest_point))
        else:
            # No projection
            return 0, point

    # Project the point back to the feasible set
    # oob_point => ouf-of-bounds point (gradient descent led to a point outside the feasible set)
    def projection(self, oob_point):
        # Some random high distance for comparison
        min_distance = 10000
        projected_point = None
        # Iterate through all the hyperplanes making up the feasible set
        for hyperplane in self.hyperplanes:
            distance, closest_point = self.vector_projection(oob_point, hyperplane)
            if distance < min_distance:
                min_distance = distance
                projected_point = closest_point
        return projected_point

    # Check for the convergence condition, i.e. [x]^{+} = x
    def convergence_check(self, previous_point, current_point):
        if previous_point is not None and current_point[0] == previous_point[0] and current_point[1] == \
                previous_point[1]:
            x_projected_point, y_projected_point = self.projection(current_point)
            # Projection of the point is the point itself
            if x_projected_point == current_point[0] and y_projected_point == current_point[1]:
                return True
        return False

    # Gradient Descent with calls to projection in case the (k+1) iteration point goes out of bounds
    def optimize(self, _step_size):
        step_size = _step_size
        if step_size is None:
            step_size = self.default_step_size
        # Previous point
        previous_point = None
        # Current point
        current_point = (0, 1)
        iteration_count = 0
        confidence = 0
        while iteration_count < self.max_iterations and ((confidence < self.confidence_bound) and (
                self.convergence_check(previous_point, current_point) is False)):
            if self.convergence_check(previous_point, current_point):
                confidence += 1
            previous_point = current_point
            self.iterations.append(iteration_count)
            self.function_values.append((current_point[0] ** 2) + (9 * (current_point[1] ** 2)))
            # Projection Gradient Descent
            current_point = self.projection(((current_point[0] - (step_size * (2 * current_point[0]))),
                                             (current_point[1] - (step_size * (18 * current_point[1])))))
            iteration_count += 1
        print('[INFO] ProjectionGradientDescent optimize: Minimization sequence completed...')

    # Visualize the convergence or divergence of the algorithm with respect to varying step sizes
    def visualize(self):
        fig, ax = plt.subplots()
        ax.plot(self.iterations, self.function_values, linewidth=1.0)
        fig.suptitle('Convergence Visualization of the Projection Gradient Descent Algorithm', fontsize=12)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.set_ylabel('Function Value', fontsize=14)
        plt.show()

    # Termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] ProjectionGradientDescent Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] ProjectionGradientDescent main: Starting system simulation...')
    projectionGradientDescent = ProjectionGradientDescent()
    projectionGradientDescent.optimize(None)
    projectionGradientDescent.visualize()
    print('[INFO] ProjectionGradientDescent main: System simulation has been stopped...')
