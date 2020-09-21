import os
import pickle
import sys

import neat
import numpy as np

import visualize

global env


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:

        print(genome.key, end="")
        if genome_id % 10 == 0:
            visualize.draw_net(config, genome, view=True)

        ffn = neat.nn.FeedForwardNetwork.create(genome, config)
        f, p, e, t = env.play(pcont=ffn)
        genome.fitness = f


if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController

    experiment_name = 'neat_evolution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # initializes environment with ai player using random controller, playing against static enemy
    env = Environment(experiment_name=experiment_name,
                      enemies=[7],
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
