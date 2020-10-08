import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns

os.environ["SDL_VIDEODRIVER"] = "dummy"

rng = np.random.default_rng()


def init_population(population_size, num_inputs, num_ouputs, num_hidden_nodes):
    individual_length = (num_inputs + 1)*num_hidden_nodes + num_ouputs*(num_hidden_nodes + 1)
    return rng.normal(size=(population_size, individual_length))


def get_fitness(individual_index, individual, env):
    fitness = evaluate(individual, env)
    result = {
        "individual_index": individual_index,
        "fitness": fitness
    }
    print(result)
    return result


def evaluate(individual, env):
    f, p, e, t = env.play(pcont=individual)
    print(f)
    return f


def generate_children(parents, num_children, mutation_rate, mutation_power):
    children = []

    for _ in range(num_children):
        # future: maybe add custom p-dist based on fitness
        parent_1, parent_2 = rng.choice(parents, size=2, replace=False)
        child = uniform_crossover(parent_1, parent_2)
        mutate(child, mutation_rate, mutation_power)
        children.append(child)

    return children


def uniform_crossover(parent_1, parent_2):
    n = len(parent_1)
    proportions = rng.random(n)
    child = parent_1 * proportions + parent_2 * (1 - proportions)
    return child


def mutate(individual, mutation_rate, mutation_power):
    n = len(individual)
    for i in range(n):
        if rng.random() < mutation_rate:
            individual[i] = rng.normal(loc=individual[i], scale=mutation_power)
    return individual


if __name__ == "__main__":
    sys.path.insert(0, 'evoman')
    from environment import Environment

    from demo_controller import player_controller

    experiment_name = 'neuro_evolution'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    df = pd.DataFrame(columns=['value', 'metric', 'gen', 'run', 'enemy_group'])

    enemy_groups = [[1, 2, 4, 7], [3, 6, 7, 8]]
    num_runs = 10
    num_hidden_nodes = 10

    # EA parameters
    max_generations = 10
    population_size = 40
    survival_rate = 0.2
    num_survivors = int(population_size * survival_rate)
    num_elites = 2
    num_children = population_size - num_elites
    mutation_rate = 0.1
    mutation_power = 1.0

    for e, enemies in enumerate(enemy_groups):

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          multiplemode='yes',
                          player_controller=player_controller(num_hidden_nodes))

        for r in range(num_runs):

            population = init_population(population_size, 20, 5, num_hidden_nodes)

            # evaluate all individuals
            for g in range(max_generations):
                generation_results = [get_fitness(i, individual, env) for i, individual in enumerate(population)]

                # sort by fitness
                sorted_results = sorted(
                    generation_results,
                    key=lambda result: result["fitness"],
                    reverse=True
                )

                winner_index = sorted_results[0]["individual_index"]
                winner = population[winner_index]
                print("best individual:", winner_index, winner)
                pickle.dump(winner, open('winners/neuro-winner-e{}-r{}'.format(e, r), 'wb'))

                df = df.append([{'value': max_fitness,
                                 'metric': 'max',
                                 'gen': gen_counter,
                                 'run': run_counter,
                                 'enemy': e
                                 },
                                {'value': mean_fitness,
                                 'metric': 'mean',
                                 'gen': g,
                                 'run': r,
                                 'enemy': e
                                 }], ignore_index=True)

                # keep elites
                elites = [population[result["individual_index"]]
                          for result in sorted_results[:num_elites]]

                # combine parents to create children
                parents = [population[result["individual_index"]]
                           for result in sorted_results[:num_survivors]]
                children = generate_children(parents, num_children, mutation_rate, mutation_power)

                # new population
                population = elites + children

    df.to_csv('neuro-results.csv', index=False)
