import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import neat
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(font_scale=1.2)

if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment
    from neat_evolution_controller import PlayerController

    experiment_name = 'test-best'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    df = pd.DataFrame(columns=['enemy', 'EA', 'fitness'])

    for ea in ['neat', 'neuro']:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config-feedforward-{}'.format(ea))

        for e in [2, 7, 8]:

            env = Environment(experiment_name=experiment_name,
                              enemies=[e],
                              player_controller=PlayerController())

            file_names = glob.glob('{}-winner-r*-e{}-*'.format(ea, e))

            file_name_best = ""
            best = 0
            for file_name in file_names:
                fitness = float(file_name.split('-')[-1])
                if fitness > best:
                    file_name_best = file_name
                    best = fitness

            winning = pickle.load(open(file_name_best, 'rb'))
            ffn = neat.nn.FeedForwardNetwork.create(winning, config)

            for r in range(5):
                f, _, _, _ = env.play(pcont=ffn)

                df = df.append({
                    'enemy': e,
                    'EA': ea,
                    'fitness': f
                }, ignore_index=True)

    sns.boxplot(data=df, x='enemy', y='fitness', hue='EA')
    sns.despine(offset=10, trim=True)
    sns.set(title='performance of best')
    plt.plot()
    plt.show()
