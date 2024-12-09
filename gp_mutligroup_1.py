import math
import random
import time
import operator
import pandas as pd
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import varAnd
from deap.tools import selRandom
from Processing_initial_dataset import read_data
import save
import os

def calEuclidean(x, y):
    x = np.array(x)
    y = np.array(y)
    dist = np.linalg.norm(x - y)
    return dist


def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def select(individuals, k, tournsize=3, fit_attr="fitness"):
    chosen = []
    for i in range(k):
        chose = []
        while len(chose) == 0:
            aspirants = selRandom(individuals, tournsize)
            max_ = 0
            for j in range(tournsize):
                if aspirants[j].fitness.values[1] >= 0:
                    if aspirants[j].fitness.values[0] > max_:
                        max_ = aspirants[j].fitness.values[0]
            for j in range(tournsize):
                if aspirants[j].fitness.values[0] == max_:
                    chose.append(aspirants[j])
        if len(chose) > 1:
            max_num = 0
            max_ = chose[0].fitness.values[1]
            for j in range(len(chose)):
                if chose[j].fitness.values[1] > max_:
                    max_num = j
                    max_ = aspirants[j].fitness.values[1]
            chosen.append(chose[max_num])
        else:
            chosen.append(chose[0])

    return chosen


def evaluate(individual, data, num, toolbox):
    feature_major = data.T
    func = toolbox.compile(expr=individual)
    pgout = func(*feature_major)
    a = calEuclidean(pgout, feature_major[num % feature_major.shape[0], :])
    b = calEuclidean(pgout, more_data[num, :])
    c = calEuclidean(feature_major[num % feature_major.shape[0], :], more_data[num, :])
    if a == 0 or b == 0:
        C = 0
    else:
        co = (c * c - a * a - b * b) / (-2 * a * b)
        if co > 1:
            co = 1
        if co < -1:
            co = -1
        C = math.degrees(math.acos(co))
    distance = b - a
    return [C, distance]

def main(i,less_data):
    SIZE = 30
    CXPB = 0.8
    MUTPB = 0.2
    LIMIT = 10
    POP_SIZE = NUMS * SIZE
    num_features = less_data.shape[1]
    pset = gp.PrimitiveSet("MAIN", num_features, 'x')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(-1, 1))
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=5)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=LIMIT))
    random.seed(i)
    np.random.seed(i)
    np.random.rand(NUMS)
    pop = toolbox.population(n=POP_SIZE)
    print('Start evolution')
    fitnesses = []
    for q in range(NUMS):
        toolbox.register("evaluate1", evaluate, data=less_data, num=q, toolbox=toolbox)
        fitnesses += list(map(toolbox.evaluate1, pop[q * SIZE:(q + 1) * SIZE]))
    # assign fitness values
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print('Evaluated %i individuals' % len(pop))
    '''The genetic operations'''
    for g in range(NGEN):
        offspring = []
        for q in range(NUMS):
            offspring_ = select(pop[q * SIZE:(q + 1) * SIZE], len(pop[q * SIZE:(q + 1) * SIZE]))
            offspring_ = varAnd(offspring_, toolbox, CXPB, MUTPB)
            offspring += offspring_
        fitnesses = []
        for q in range(NUMS):
            toolbox.register("evaluate2", evaluate, data=less_data, num=q, toolbox=toolbox)
            fitnesses += list(map(toolbox.evaluate2, offspring[q * SIZE:(q + 1) * SIZE]))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        print('Evaluated %i individuals' % len(offspring))
        pop[:] = offspring
        print(pop)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        record = stats.compile(pop)
        logbook = tools.Logbook()
        logbook.record(gen=g, evals=30, **record)
        logbook.header = "gen", "avg", "min", "max"
        print(logbook)

    print("-- End of (successful) evolution --")
    best_ind = []
    for q in range(NUMS):
        best_ind += tools.selBest(pop[q * SIZE:(q + 1) * SIZE], 1)[0]
    result = np.zeros(shape=(NUMS, less_data.shape[0]))
    result_f = pd.DataFrame(columns=['expression'])
    resultall, resultall_f, result, result_f = save.store_excel(result, result_f, toolbox, pop, less_data, NUMS, SIZE, POP_SIZE, less_data)

    print(result)
    print(i)
    return resultall, resultall_f, result, result_f

if __name__ == "__main__":
    folder_path = 'data'  #储存数据文件夹
    file_names = os.listdir(folder_path)
    NGEN = 100
    for file in file_names:
        data_name = 'data\\' + file
        less_data, more_data, NUMS, less_class, more_class, Train = read_data(data_name, 0)
        less_data = less_data.to_numpy()
        more_data = more_data.to_numpy()
        less_data = less_data.T
        for i in range(30):
            b_time = time.time()
            resultall, resultall_f, result, result_f = main(i,less_data,)
            n_time = time.time()
            save.save_allproduce(resultall, i, file, NGEN)
            save.save_allexpression(resultall, resultall_f, i, b_time, n_time, file, NGEN)
            save.save_expression(result, result_f, i, b_time, n_time, file, NGEN)
            labels_np = np.full((result.shape[0], 1), less_class)
            result = np.hstack((result, labels_np))
            result = np.vstack((result, Train))
            save.save_produce(result, i, file, NGEN)