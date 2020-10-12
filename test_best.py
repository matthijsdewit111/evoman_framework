import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import neat
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)
mpl.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController
    from demo_controller import player_controller

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward-neat')

    experiment_name = 'test-best'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    df = pd.DataFrame(columns=['enemy', 'EA', 'gain'])

    determine_winners = False
    winners_neat = [(9, 34.0), (1, 10.0)]
    winners_neuro = [(2, 34.0), (3, 16.0)]

    for e in range(2):
        print("e:", e)
        if determine_winners:
            for r in range(10):
                genome_neuro = pickle.load(open('winners/neuro-winner-e{}-r{}'.format(e, r), 'rb'))
                genome_neat = pickle.load(open('winners/neat-winner-e{}-r{}'.format(e, r), 'rb'))
                ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)

                total_gain_neat = 0
                total_gain_neuro = 0
                for enemy in range(1, 9):
                    env_neat = Environment(experiment_name=experiment_name,
                                        enemies=[enemy],
                                        player_controller=PlayerController())

                    _, pe_neat, ee_neat, _ = env_neat.play(pcont=ffn)
                    total_gain_neat += pe_neat - ee_neat

                    env_neuro = Environment(experiment_name=experiment_name,
                                            enemies=[enemy],
                                            player_controller=player_controller(10))

                    _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                    total_gain_neuro += pe_neuro - ee_neuro

                    if winners_neat[e][1] < total_gain_neat:
                        winners_neat[e] = (r, total_gain_neat)
                    if winners_neuro[e][1] < total_gain_neuro:
                        winners_neuro[e] = (r, total_gain_neuro)

        print(winners_neat)
        print(winners_neuro)

        genome_neuro = pickle.load(open('winners/neuro-winner-e{}-r{}'.format(e, winners_neuro[e][0]), 'rb'))
        genome_neat = pickle.load(open('winners/neat-winner-e{}-r{}'.format(e, winners_neat[e][0]), 'rb'))
        ffn = neat.nn.FeedForwardNetwork.create(genome_neat, config)

        for v in range(5):
            print("v:", v)
            total_gain_neat = 0
            total_gain_neuro = 0
            for enemy in range(1, 9):
                env_neat = Environment(experiment_name=experiment_name,
                                       enemies=[enemy],
                                       player_controller=PlayerController())

                _, pe_neat, ee_neat, _ = env_neat.play(pcont=ffn)
                total_gain_neat += pe_neat - ee_neat

                env_neuro = Environment(experiment_name=experiment_name,
                                        enemies=[enemy],
                                        player_controller=player_controller(10))

                _, pe_neuro, ee_neuro, _ = env_neuro.play(pcont=genome_neuro)
                total_gain_neuro += pe_neuro - ee_neuro

            df = df.append([{
                'enemy': e,
                'EA': 'neat',
                'gain': total_gain_neat
            }, {
                'enemy': e,
                'EA': 'neuro',
                'gain': total_gain_neuro
            }], ignore_index=True)

    print(winners_neat)
    print(winners_neuro)
    df.to_csv('results-winner-gains.csv', index=False)

    sns.boxplot(data=df, x='enemy', y='gain', hue='EA').set_title('gain of best solutions')
    sns.despine(offset=10, trim=True)

    plt.tight_layout()
    plt.savefig('gains.png')
