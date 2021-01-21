#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 20:58:05 2020

@author: z fang
"""

import pickle
import random
from graph_tool import *
import numpy as np
from math import exp
from haversine import haversine
from graph_tool.topology import shortest_path

N = 20*1000 #Number of cars
l = 2 #Number of lanes
rc = 22.1 #critical density
v_jam = 8 #jam speed

# load road network
road_network = load_graph("./result/road_network.xml.gz")
# load movements frequency
# load movements frequency of autos
with open("result/movements_frequency_autos.pkl","rb") as f:
    movements_frequency_autos = pickle.load(f)

with open("result/dict_centroids_projected.pkl","rb") as f:
    dict_centroids_projected = pickle.load(f)


with open("result/edge_densities_new.pkl","rb") as f:
    
    edge_densities_new = pickle.load(f)
    
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



def generate_weights0(edge_densities, rc=22.1):
    #forcus on high densities
    weights = []
    for i in range(len(edge_densities)):
        if edge_densities[i]<=2*rc:
            weights.append(0)
        else:
            weights.append(random.uniform(0,edge_densities[i]/(2*rc)))
            
    
    weights = (np.array(weights)/sum(weights)).tolist()
    return weights


def generate_weights1(edge_densities, rc=22.1):
    w = []
    for d in edge_densities:
        if d==0:
            w.append(-0.01)
        elif d< 22.1:
            w.append(random.uniform(-(rc/d),0))
        else:
            w.append(0)

    w_abs = [abs(wi) for wi in w]
    w = (np.array(w)/sum(w_abs)).tolist()    
    return w


# objective function
def simulation_shortest_redistribution_new_inits(w):
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
    edge_maxspeed = road_network.edge_properties["fakespeed"]
    edge_hdistance = road_network.edge_properties["hdistance"]
    # create new property map for total weights
    edge_weight = road_network.new_edge_property("double")
    #build a dictionary to store number cars on edges
    edges = list(road_network.edges())
    for edge,wi in list(zip(edges,w)):
        edge_weight[edge]=max(wi+edge_hdistance[edge],0) #to avoid negative weights
    dict_edge_cars = dict(zip(edges,[0]*len(edges)))
    
    #dict_movement_path = dict()
    for mc in movements_counts:
        m = mc[0]
        m_p = (dict_centroids_projected[m[0]], dict_centroids_projected[m[1]])
        m_count = mc[1]
        #compute nodes and shortest path
        nodes, path = shortest_path(road_network,m_p[0],m_p[1],weights=edge_weight,
                                   negative_weights=False)
        distance = sum([edge_hdistance[e] for e in path])
        if distance!=0:
            r_path = m_count/distance #density of path
            #dict_movement_path[m] = path
            for edge in path:
                dict_edge_cars[edge]+=edge_hdistance[edge]*r_path
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
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

import random
from deap import tools

IND_SIZE = road_network.num_edges()

toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, -4,4)
#toolbox.register("attribute0", random.uniform, 0,0)
#toolbox.register("generate_w0", creator.Individual,generate_weights0, edge_densities, 22.1)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                toolbox.attribute, n=IND_SIZE)
#toolbox.register("individual_e0", creator.Individual, toolbox.generate_w0)
#toolbox.register("individual0", tools.initRepeat, creator.Individual,
#                 toolbox.attribute0, n=IND_SIZE)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", simulation_shortest_redistribution_new_inits)
toolbox.register("mate", tools.cxOnePoint) #crossover
toolbox.register("mutate", tools.mutGaussian) #Gaussian mutation
toolbox.register("mat_select", tools.selTournament, tournsize=5)
toolbox.register("environ_select", tools.selBest)

def optimize_redistribution_new_inits():
    print("Initialization")
    pop = []
    ind0 = creator.Individual([0]*IND_SIZE)
    pop.append(ind0)
    for i in range(5):
        indi = creator.Individual(generate_weights0(edge_densities_new))
        pop.append(indi)
    for j in range(4):
        indj = creator.Individual(generate_weights1(edge_densities_new))
        pop.append(indj)

    fitnesses = list(map(toolbox.evaluate, pop))
    print("Initial Evaluation: ", fitnesses)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    CXPB = 0.7
    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations
    g = 0
    
    minfits = []
    meanfits = []
    maxfits = []
    
    minfits.append(min(fits))
    meanfits.append(sum(fits)/len(fits))
    maxfits.append(max(fits))
    
    
    
    # Begin the evolution
    while g < 11:
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
            toolbox.mutate(mutant,0, abs((sum(mutant)+1)/len(mutant)), 0.2)
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
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        
        minfits.append(min(fits))
        meanfits.append(max(fits))
        maxfits.append(mean)
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
    with open("minfits_inits_new_fakespeed.pkl","wb") as f:
        pickle.dump(minfits, f)
    
    with open("meanfits_inits_new_fakespeed.pkl","wb") as f:
        pickle.dump(meanfits, f)
        
    with open("maxfits_inits_new_fakespeed.pkl","wb") as f:
        pickle.dump(maxfits, f)

    return minfits,meanfits,maxfits


optimize_redistribution_new_inits()








