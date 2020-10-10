import os
import pickle
import sys

import numpy as np
import pandas as pd
import seaborn as sns

os.environ["SDL_VIDEODRIVER"] = "dummy"

rng = np.random.default_rng()


def init_population(population_size, num_inputs, num_ouputs, num_hidden_nodes, initial_species):
    individual_length = (num_inputs + 1)*num_hidden_nodes + num_ouputs*(num_hidden_nodes + 1)
    population = rng.normal(size=(population_size, individual_length))
    return np.array_split(population, initial_species)


def get_fitness(individual_index, species_index, individual, env):
    fitness = evaluate(individual, env)
    result = {
        "individual_index": individual_index,
        "species_index": species_index,
        "fitness": fitness
    }
    print(result)
    return result


def evaluate(individual, env):
    f, p, e, t = env.play(pcont=individual)
    return f


def generate_children(parents, num_children, mutation_params):
    children = []

    for _ in range(num_children):
        # future: maybe add custom p-dist based on fitness
        parent_1, parent_2 = rng.choice(parents, size=2, replace=False)
        child = uniform_crossover(parent_1, parent_2)
        mutate(child, mutation_params)
        children.append(child)

    return children


def uniform_crossover(parent_1, parent_2):
    proportions = rng.random(len(parent_1))
    child = parent_1 * proportions + parent_2 * (1 - proportions)
    return child


def mutate(individual, mutation_params):
    mutation_replace_rate = mutation_params["mutation_replace_rate"]
    mutation_rate = mutation_params["mutation_rate"]
    mutation_power = mutation_params["mutation_power"]

    for i in range(len(individual)):
        if rng.random() < mutation_replace_rate:
            individual[i] = rng.normal()
        else:
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
    max_generations = 20
    population_size = 50
    
    # per species
    survival_rate = 0.2
    num_elites = 1
    
    initial_species = 10
    min_species = 3
    min_species_size = 2
    max_stagnation = 10
    
    mutation_params = {
        "mutation_rate": 0.1,
        "mutation_power": 1.0,
        "mutation_replace_rate": 0.1
    }

    for e, enemies in enumerate(enemy_groups):

        env = Environment(experiment_name=experiment_name,
                          enemies=enemies,
                          multiplemode='yes',
                          player_controller=player_controller(num_hidden_nodes))

        for r in range(num_runs):

            population = init_population(population_size, 20, 5, num_hidden_nodes, initial_species)
            num_species = initial_species
            stagnation_array = [[0, 0]] * num_species # [stagnation counter, max_fitness]
            winner = [None, 0] # [geneome, fitness]

            # evaluate all individuals
            for g in range(max_generations):
                
                generation_results = []
                elites = []
                parents = []
                max_fitnesses = []

                for s, sub_population in enumerate(population):
                    species_generation_results = [get_fitness(i, s, individual, env) for i, individual in enumerate(sub_population)]
                    generation_results.append(species_generation_results)

                    # sort by fitness
                    sorted_results = sorted(
                        species_generation_results,
                        key=lambda result: result["fitness"],
                        reverse=True
                    )

                    species_max_fitness = sorted_results[0]["fitness"]
                    max_fitnesses.append(species_max_fitness)
                    if species_max_fitness > winner[1]:
                        winner_index = sorted_results[0]["individual_index"]
                        winner = [sub_population[winner_index], species_max_fitness]
                    
                    # keep elites
                    species_elites = [sub_population[result["individual_index"]]
                        for result in sorted_results[:num_elites]]
                    elites.append(species_elites)
                    
                    # determine parents which will create children
                    num_survivors = int(len(sub_population) * survival_rate)
                    num_children = len(sub_population) - num_elites
                    species_parents = [sub_population[result["individual_index"]]
                            for result in sorted_results[:num_survivors]]
                    parents.append(species_parents)

                max_fitness = 0
                sum_fitnesses = 0
                for i in range(num_species):
                    species_max_fitness = 0
                    sum_species_fitness = 0
                    for j in range(len(population)):
                        fitness = generation_results[i][j]["fitness"]
                        if fitness > max_fitness:
                            max_fitness = fitness
                        if fitness > species_max_fitness:
                            species_max_fitness = fitness
                        sum_fitnesses += fitness
                        sum_species_fitness += fitness
                    
                    print("species {}: max fit {}, mean fit {}, stagnation {}".format(species_max_fitness, sum_species_fitness / len(population), stagnation_array[i]))

                mean_fitness = sum_fitnesses / sum([len(sp) for sp in population])
                print("overal: max fit: {}, mean fit {}".format(max_fitness, mean_fitness))

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
                
                # keep track of stagnation
                stagnated_species = []
                for i in range(num_species):
                    if stagnation_array[i][1] < max_fitnesses[i]:
                       stagnation_array[i][1] = max_fitnesses[i]
                       stagnation_array[i][0] = 0
                    else:
                        stagnation_array[i][0] += 1
                        if stagnation_array[i][0] > max_stagnation:
                            stagnated_species.append(i)
                            print("species", i, "stagnated")
                
                # delete stagnated species
                population_lost = 0
                for i in stagnated_species:
                    num_species -= 1
                    population_lost += len(population[i])
                    del population[i]
                    del stagnation_array[i]
                    del elites[i]
                    del parents[i]
                    del max_fitnesses[i]
                
                # grow other species based on their fitness to keep total population the same
                for _ in range(population_lost):
                    # choose a random sub population based on their population fitness
                    sub_population = rng.choice(population, p=np.array(max_fitnesses)/sum(max_fitnesses))
                    sub_population.append(None) # just a placeholder to keep track of their size


                # new population
                for i in range(num_species):
                    num_children = len(population[i]) - num_elites
                    children = generate_children(parents[i], num_children, mutation_params)
                    population[i] = elites[i] + children
            
        print("winner:", winner)
        pickle.dump(winner, open('winners/neuro-winner-e{}-r{}'.format(e, r), 'wb'))

    df.to_csv('neuro-results.csv', index=False)
