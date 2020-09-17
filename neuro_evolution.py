import sys
import os
sys.path.insert(0, 'evoman')
from neuro_evolution_controller import PlayerController
from environment import Environment
from neural_network import NeuralNetwork
import numpy as np

experiment_name = 'neuro_evolution'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

max_no_generation = 5
pop_size = 5
no_children = 5

# 20 always input, 5 always output
layer_sizes = [20, 10, 5]
def init_nn():
    # 1 + [0, inf] layers, probability exponential decreasing, scale = expected/mean value
    # num_hidden_layers = 1 + round(np.random.exponential(scale=1))
    # 1 + [0, inf] nodes per layer, probability exponential decreasing
    # hidden_layers = [1 + round(np.random.exponential(scale=4)) for _ in range(num_hidden_layers)]

    # layer_sizes = [20] + hidden_layers + [5]
    weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
    weights = [np.random.standard_normal(s) / np.sqrt(s[1]) for s in weight_shapes]
    biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    return NeuralNetwork(layer_sizes, weights, biases)

population = [init_nn() for i in range(pop_size)]

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[2],
                  playermode="ai",
                  player_controller=PlayerController(),
                  enemymode="static",
                  level=1,
                  speed="fastest")

population_fitness = np.zeros(pop_size)
for generation in range(max_no_generation):
    print("GENERATION:", generation)

    # run simulations
    for i in range(pop_size):
        f, p, e, t = env.play(pcont=population[i])
        population_fitness[i] = f

    # parent selection, select 2 best parents
    index_p1, index_p2 = (-population_fitness).argsort()[:2]

    p1 = population[index_p1]
    p2 = population[index_p2]

    children = []
    for i in range(no_children):
        # crossover
        crossover_bias = np.random.random()
        child_weights = np.array(p1.weights) * crossover_bias + (1 - crossover_bias) * np.array(p2.weights)
        child_biases = np.array(p1.biases) * crossover_bias + (1 - crossover_bias) * np.array(p2.biases)

        # mutations TODO

        child = NeuralNetwork(layer_sizes, child_weights, child_biases)
        children.append(child)
    
    # replace parents

    indices_worst = (population_fitness).argsort()[:no_children]

    for i, j in enumerate(indices_worst):
        population[j] = children[i]
