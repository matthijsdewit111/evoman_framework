import os
import pickle
import sys

import neat
import numpy as np
import pandas as pd
import seaborn as sns

import visualize

# global vars
df = pd.DataFrame(columns=['value', 'metric', 'gen', 'run', 'enemy'])
run_counter = 0
gen_counter = 0
enemy = 0


def eval_genomes(genomes, config):
    global gen_counter, df

    fitnesses = np.zeros(len(genomes))
    for i, (genome_id, genome) in enumerate(genomes):

        print(i, genome.key, end="")
        # if i % 10 == 0:
        #     visualize.draw_net(config, genome, view=True)

        ffn = neat.nn.FeedForwardNetwork.create(genome, config)
        f, p, e, t = env.play(pcont=ffn)
        genome.fitness = f
        fitnesses[i] = f

    max_fitness = fitnesses.max()
    mean_fitness = fitnesses.mean()

    df = df.append([{'value': max_fitness,
                     'metric': 'max',
                     'gen': gen_counter,
                     'run': run_counter,
                     'enemy': enemy
                     },
                    {'value': mean_fitness,
                     'metric': 'mean',
                     'gen': gen_counter,
                     'run': run_counter,
                     'enemy': enemy
                     }], ignore_index=True)

    gen_counter += 1


if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController

    experiment_name = 'neuro_evolution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    enemies = [2, 7, 8]
    num_runs = 10

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-neuro')

    for e in enemies:
        enemy = e

        env = Environment(experiment_name=experiment_name,
                          enemies=[e],
                          player_controller=PlayerController())

        for r in range(num_runs):
            gen_counter = 0
            run_counter = r

            p = neat.Population(config)
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix='checkpoints/neuro-checkpoint-e{}-r{}'.format(e, r)))

            winner = p.run(eval_genomes, 25)
            pickle.dump(winner, open('neuro-winner-r{}-e{}-{}'.format(r, e, round(winner.fitness, 3)), 'wb'))

            # print('\nBest genome:\n{!s}'.format(winner))
            # visualize.draw_net(config, winner, view=True)

    df.to_csv('neuro-results.csv', index=False)
    # while True:
    #     input("Press Enter to watch it play...")

    #     env.speed = "normal"

    #     ffn = neat.nn.FeedForwardNetwork.create(winner, config)
    #     env.play(pcont=ffn)
