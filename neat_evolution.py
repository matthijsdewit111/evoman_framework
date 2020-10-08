import os
import pickle
import sys

import neat
import numpy as np
import pandas as pd
import seaborn as sns

import visualize

os.environ["SDL_VIDEODRIVER"] = "dummy"

# global vars
df = pd.DataFrame(columns=['value', 'metric', 'gen', 'run', 'enemy_group'])
run_counter = 0
gen_counter = 0
enemy_group = 0


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
                     'enemy_group': enemy_group
                     },
                    {'value': mean_fitness,
                     'metric': 'mean',
                     'gen': gen_counter,
                     'run': run_counter,
                     'enemy_group': enemy_group
                     }], ignore_index=True)

    gen_counter += 1


if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController

    experiment_name = 'neat_evolution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    enemy_groups = [[1,2,4,7], [3,6,7,8]]
    num_runs = 10

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-neat')

    for e, enemies in enumerate(enemy_groups):
        enemy_group = e

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          multiplemode='yes',
                          player_controller=PlayerController())

        for r in range(num_runs):
            gen_counter = 0
            run_counter = r

            p = neat.Population(config)
            p.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            p.add_reporter(stats)
            p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix='checkpoints/neat-checkpoint-e{}-r{}'.format(e, r)))

            winner = p.run(eval_genomes, 25)
            pickle.dump(winner, open('winners/neat-winner-e{}-r{}'.format(e, r), 'wb'))

    df.to_csv('neat-results.csv', index=False)
