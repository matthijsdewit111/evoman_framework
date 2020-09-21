import os
import pickle
import sys

import neat
import numpy as np
import matplotlib.pyplot as plt

import visualize

plt.ion()

counter = 0
previous_fitness_max = 0.0
previous_fitness_mean = 0.0
previous_fitness_min = 0.0
previous_fitness_std = 0.0


def eval_genomes(genomes, config):
    global counter, previous_fitness_mean, previous_fitness_std, previous_fitness_max, previous_fitness_min
    counter += 1
    fitnesses = []
    for genome_id, genome in genomes:

        print(genome.key, end="")
        # if counter % 10 == 0:
        #     visualize.draw_net(config, genome, view=True)

        ffn = neat.nn.FeedForwardNetwork.create(genome, config)
        f, p, e, t = env.play(pcont=ffn)
        genome.fitness = f
        fitnesses.append(f)

    np_fitnesses = np.array(fitnesses)
    fitness_mean = np_fitnesses.mean()
    fitness_std = np_fitnesses.std()
    fitness_max = np_fitnesses.max()
    fitness_min = np_fitnesses.min()

    x = [counter - 1, counter]

    plt.plot(x, [previous_fitness_mean, fitness_mean], color='blue')
    plt.fill_between(x,
                     [previous_fitness_mean - previous_fitness_std, fitness_mean - fitness_std],
                     [previous_fitness_mean + previous_fitness_std, fitness_mean + fitness_std],
                     color='blue', alpha=0.5, linewidth=0.0)
    plt.plot(x, [previous_fitness_max, fitness_max], color='green')
    plt.plot(x, [previous_fitness_min, fitness_min], color='red')
    plt.pause(0.1)

    previous_fitness_mean = fitness_mean
    previous_fitness_std = fitness_std
    previous_fitness_min = fitness_min
    previous_fitness_max = fitness_max


if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController

    experiment_name = 'neat_evolution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                      enemies=[8],
                      playermode="ai",
                      player_controller=PlayerController(),
                      enemymode="static",
                      speed="fastest")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 300)
    pickle.dump(winner, open('neat-winner', 'wb'))

    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, view=True)

    while True:
        input("Press Enter to watch it play...")

        env.speed = "normal"

        ffn = neat.nn.FeedForwardNetwork.create(winner, config)
        env.play(pcont=ffn)
