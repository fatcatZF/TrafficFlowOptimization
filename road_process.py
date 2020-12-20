#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:08:06 2020
@author: Z Fang
"""

import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint
import matplotlib.pyplot as plt
import numpy as np

#path of road data
road_file = "berlin-latest-free/gis_osm_roads_free_1.shp"
#read road data
gdf_road = gpd.read_file(road_file)

print("Number of roads: ", len(gdf_road))

#define condition to filter roads
for_cars = ((gdf_road.fclass=="motorway")|(gdf_road.fclass=="trunk")|\
           (gdf_road.fclass=="primary")|(gdf_road.fclass=="secondary")|(gdf_road.fclass=="tertiary"))&\
            (gdf_road.maxspeed>=10)
            
gdf_road_cars = gdf_road[for_cars]
print("Number of filtered roads: ", len(gdf_road_cars))

def get_intersections(lines):
    """
    """
    point_intersections = []
    line_intersections = []
    lines_len = len(lines)
    for i in range(lines_len-1):
        for j in range(i+1,lines_len):
            l1,l2 = lines[i],lines[j]
            if l1.intersects(l2):
                intersection=l1.intersection(l2)
                if isinstance(intersection, LineString):
                     line_intersections.append(intersection)
                     inter_list = list(intersection.coords)
                     inter_list_points = [Point(p) for p in inter_list]
                     point_intersections+=inter_list_points
                elif isinstance(intersection, Point):
                     point_intersections.append(intersection)
                elif isinstance(intersection, MultiPoint):
                    points_ij = [Point(p.x,p.y) for p in intersection]
                    point_intersections+=points_ij
    return point_intersections, line_intersections
                  
            
