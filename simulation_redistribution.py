#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:14:55 2020

@author: z fang
"""

import pickle
from graph_tool import *
import numpy as np
from math import exp
import pandas as pd
from haversine import haversine
from graph_tool.topology import shortest_path

N = 20*1000 #Number of cars
l = 2 #Number of lanes
rc = 22.1
v_jam = 8

# load road network
road_network = load_graph("./result/road_network.xml.gz")
# load movements frequency
# load movements frequency of autos
with open("result/movements_frequency_autos.pkl","rb") as f:
    movements_frequency_autos = pickle.load(f)

with open("result/dict_centroids_projected.pkl","rb") as f:
    dict_centroids_projected = pickle.load(f)


def compute_edgetime(r, d,vm, rc=rc, v_jam=v_jam):
    """
    args: 
       r: density
       d: haversine distance of the edge
       vm: maxspeed of edge
       rc: critical density(22.1)
       v_jam: jam velocity
    return: time of a car spending on this edge
    """
    fr = (vm/rc)*max(0,r*(2*rc-r)) #computing flow
    vr = max(v_jam, min(vm, fr/r)) #computing velocity
    tr = d/vr #computing time
    return tr




# objective function
def simulation__shortest_redistribution(w):
    """
    # objective function
    args:
     w: a vector contains imposed weights for edges
    """
    #extract movements & frequencies
    movements = [m[0] for m in movements_frequency_autos]
    frequencies = np.array([m[1] for m in movements_frequency_autos])
    #computing counts based on Ni and frequencies
    counts = frequencies*N
    #build movements and counts list
    movements_counts = list(zip(movements, counts))
    # create edge property map
    edge_maxspeed = road_network.edge_properties["maxspeed"]
    edge_hdistance = road_network.edge_properties["hdistance"]
    # create new property map for total weights
    edge_weight = road_network.new_edge_property("double")
    #build a dictionary to store number cars on edges
    edges = list(road_network.edges())
    for edge,wi in list(zip(edges,w)):
        edge_weight[edge]=max(wi+edge_hdistance[edge],0) #to avoid negative weights
    dict_edge_cars = dict(zip(edges,[0]*len(edges)))
    #build a dictionary to store shortest path of movement
    #dict_movement_path = dict()
    for mc in movements_counts:
        m = mc[0]
        m_p = (dict_centroids_projected[m[0]], dict_centroids_projected[m[1]])
        m_count = mc[1]
        #compute nodes and shortest path
        nodes, path = shortest_path(road_network,m_p[0],m_p[1],weights=edge_weight,
                                   negative_weights=False)
        #dict_movement_path[m] = path
        for edge in path:
            dict_edge_cars[edge]+=m_count 
    #build a dictionary to store the time of edges
    dict_edge_time = dict(list(zip(edges,[0]*len(edges))))
    #compute time on edges
    t_tot = 0
    for edge in edges:
        hdistance = edge_hdistance[edge] #haversine distance of the edge
        r_edge = dict_edge_cars[edge]/(hdistance*l) #density of edge
        if r_edge == 0:continue
        maxspeed = edge_maxspeed[edge]
        t_edge = compute_edgetime(r_edge,hdistance, maxspeed)
        dict_edge_time[edge] = t_edge
        t_tot += t_edge*dict_edge_cars[edge]
    
    #dict_movement_time = dict()
    
    #for m in movements:
        #path = dict_movement_path[m]
        #m_t = sum([dict_edge_time[e] for e in path])
        #dict_movement_time[m] = m_t
        
    return t_tot,


from deap import base, creator
import random
from deap import tools

IND_SIZE = road_network.num_edges()

try:
    del creator.FitnessMin
except:
    print("No this class")
    
try:
    del creator.Individual
except:
    print("No this class")
    

creator.create("FitnessMin", base.Fitness, weights=(-1,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

try:
    toolbox.unregister("attribute")
except:
    pass

try:
    toolbox.unregister("attribute0")
except:
    pass        

try:
    toolbox.unregister("individual")
except:
    pass  

try:
    toolbox.unregister("individual0")
except:
    pass  

try:
    toolbox.unregister("population")
except:
    pass  

try:
    toolbox.unregister("evaluate")
except:
    pass  

try:
    toolbox.unregister("mate")
except:
    pass  


try:
    toolbox.unregister("mutate")
except:
    pass  

try:
    toolbox.unregister("mat_select")
except:
    pass  

try:
    toolbox.unregister("environ_select")
except:
    pass  

toolbox.register("attribute", random.random)
toolbox.register("attribute0", random.uniform, 0,0)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)
toolbox.register("individual0", tools.initRepeat, creator.Individual,
                 toolbox.attribute0, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", simulation__shortest_redistribution)
toolbox.register("mate", tools.cxOnePoint) #crossover
toolbox.register("mutate", tools.mutGaussian) #Gaussian mutation
toolbox.register("mat_select", tools.selTournament, tournsize=5)
toolbox.register("environ_select", tools.selBest)

def optimize_redistribute():
    print("Initialization")
    pop = toolbox.population(n=9)
    ind0 = toolbox.individual0()
    pop.append(ind0)

    fitnesses = list(map(toolbox.evaluate, pop))
    print("Initial Evaluation: ", fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    CXPB = 0.7
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0
    
    max_fitnesses = [] #record maximal fitnesses of every generation
    mean_fitnesses = [] #record mean fitnesses of every generation
    min_fitnesses = [] #record minimal fitnesses of every generation
    min_fitnesses.append(min(fits))
    max_fitnesses.append(max(fits))
    mean_fitnesses.append(sum(fits)/len(fits))
    
    # Begin the evolution
    while g < 10:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.mat_select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            #print(mutant)
            toolbox.mutate(mutant,0, abs(sum(mutant)/len(mutant)), 0.2)
            del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        new_pop = toolbox.environ_select(pop+offspring, len(pop))
        pop[:] = new_pop
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        min_fitnesses.append(min(fits))
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        min_fitnesses.append(min(fits))
        mean_fitnesses.append(max(fits))
        max_fitnesses.append(mean)
                
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    
    with open("min_fitnesses_redistribution.pkl", "wb") as f:
        pickle.dump(min_fitnesses, f)

    with open("max_fitnesses_redistribution.pkl","wb") as f:
        pickle.dump(max_fitnesses, f)

    with open("mean_fitnesses_redistribution.pkl","wb") as f:
        pickle.dump(mean_fitnesses, f)
        
    return min_fitnesses, mean_fitnesses, max_fitnesses


optimize_redistribute()


