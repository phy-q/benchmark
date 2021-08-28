#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:22:08 2019

@author: chengxue
"""
import numpy as np
#from shapely.geometry import MultiPolygon, Polygon, box
#from shapely.ops import unary_union

class Rectangle:
    def __init__(self, *args, **kwargs):
        '''
        points need in tuple with 2 arrays,
        the first array represents the y value of the points
        the second array represents the x value of the points
        '''
        self.points = None
        self.bottom_right = None
        self.top_left = None
        self.height = None
        self.width = None

        #top left X and Y
        self.X = None
        self.Y = None

        rectangle = kwargs.get('rectangle',None)
        if isinstance(rectangle, Rectangle):
            self.points = rectangle.points
            self.bottom_right = rectangle.bottom_right
            self.top_left = rectangle.top_left
            self.X = self.top_left[0]
            self.Y = self.top_left[1]
            diff = self.bottom_right - self.top_left
            self.height = diff[0]
            self.width = diff[1]

        elif len(args) > 0:
            self.points = args[0]
            self.bottom_right =  np.max(self.points,1)[::-1]
            self.top_left = np.min(self.points,1)[::-1]
            self.X = self.top_left[0]
            self.Y = self.top_left[1]
            diff = self.bottom_right - self.top_left
            self.height = diff[0]
            self.width = diff[1]

    #centre point

    def get_centre_point(self):
        '''
        get the centre point for each bounding box
        '''
        return np.array([self.top_left[1]+ self.height/2,self.top_left[0]+ self.width/2])


    def add(self,other):
        '''
        inputs a rectangle object
        updates the top_left,bottom_right, height and width
        '''

        self.bottom_right[0] = max(other.bottom_right[0],self.bottom_right[0])
        self.bottom_right[1] = max(other.bottom_right[1],self.bottom_right[1])

        self.top_left[0] = min(other.top_left[0],self.top_left[0])
        self.top_left[1] = min(other.top_left[1],self.top_left[1])
        self.X = self.top_left[0]
        self.Y = self.top_left[1]

        diff = self.bottom_right - self.top_left
        self.height = diff[1]
        self.width = diff[0]

    def intersects(self, other):
        '''
        check if two boxes intersect
        '''
        min_x,min_y = self.top_left
        max_x,max_y = self.bottom_right
        o_min_x,o_min_y = other.top_left
        o_max_x,o_max_y = other.bottom_right


        if (max_x < o_min_x or o_max_x < min_x or max_y < o_min_y or o_max_y < min_y):
            return False
        else:
            return True

    def dialate(self,dx,dy):
        '''
        enlarge the bounding box by dx and dy
        '''
        self.top_left[1] = self.top_left[1] - dy
        self.top_left[0] = self.top_left[0] - dx
        self.X = self.top_left[0]
        self.Y = self.top_left[1]

        self.bottom_right[1] = self.bottom_right[1] + 2 * dy
        self.bottom_right[0] = self.bottom_right[0] + 2 * dx




    def check_val(self, width, height):

        '''
        check if the bounding box exceeds the boundary
        '''

        if self.top_left[0] < 0:
            self.top_left[0] = 0
            self.X = self.top_left[0]

        if self.top_left[1] > width - 1:
            self.top_left[1] = width - 1
            self.Y = self.top_left[1]

        if self.bottom_right[0] < 0:
            self.bottom_right[0] = 0
        if self.bottom_right[1] > height - 1:
            self.bottom_right[1] = height - 1


#def platformCombiner(parsed_json):
#    '''
#    This function combines any platforms that are intersecting with each other to one
#    platform and return the combined list of all platforms.
#    
#    @input parsed_json : a list of json records
#    @output: a list of json records with platform vertices
#    '''
#    #Obtaining sets of seperately connected polygons
#    all_vert = []
#    
#    for item in parsed_json:
#        if item["colormap"][0:24] == "{color: 36,percent: 0.55":
#            poly = Polygon([(item["vertices"][0]["x"],item["vertices"][0]["y"]),
#                            (item["vertices"][1]["x"],item["vertices"][1]["y"]),
#                            (item["vertices"][2]["x"],item["vertices"][2]["y"]),
#                            (item["vertices"][3]["x"],item["vertices"][3]["y"])])
#            if poly.is_valid:
#    
#                all_vert.append(poly)  
#            
#    sep_vert = unary_union(all_vert)
#    if sep_vert.geom_type == 'Polygon':
#        sep_vert = MultiPolygon([sep_vert])
#        
#    plat_list = []
#    #Obtaining bounds of the polygons if they contain similar x and y values 
#    for platform in sep_vert:
#        temp_dict = {}
#        temp_dict['type'] = "Platform"
#        temp_dict['vertices'] = []
#        l1 = list(platform.exterior.coords)
#        x = set(list(zip(*list(platform.exterior.coords)))[0]) # unique x elements in the list of seperate platforms
#        y = set(list(zip(*list(platform.exterior.coords)))[1]) # unique y elements in the list of seperate platforms
#        
#        if len(x) >= len(y):
#            for i in range(len(l1) - 1):
#                x1, y1 = l1[i]
#                x2, y2 = l1[i + 1]
#                if y1 == y2:
#                    continue
#                else:
#                    temp_dict['vertices'].append({"x": x1, "y": y1})
#                    temp_dict['vertices'].append({"x": x2, "y": y2})
#                    
#        else:
#            for i in range(len(l1) - 1):
#                x1, y1 = l1[i]
#                x2, y2 = l1[i + 1]
#                if x1 == x2:
#                    continue
#                else:
#                    temp_dict['vertices'].append({"x": x1, "y": y1})
#                    temp_dict['vertices'].append({"x": x2, "y": y2})
#                    
#        plat_list.append(temp_dict)
#        
#    return plat_list


# def platformCombiner(parsed_json):
    
#     L= []

#     for item in parsed_json:
#         if item["type"] == "Platform":
#             poly = Polygon([(item["vertices"][0]["x"],item["vertices"][0]["y"]),(item["vertices"][1]["x"],item["vertices"][1]["y"]),(item["vertices"][2]["x"],item["vertices"][2]["y"]),(item["vertices"][3]["x"],item["vertices"][3]["y"])])
#             L.append(poly)  

#     P = unary_union(L)

#     if P.geom_type == 'Polygon':
#         P = MultiPolygon([P])
    
    
#     plat_dict = []

#     for Q in P:
#         temp_dict = {}
#         temp_dict['type'] = "Platform"
#         temp_dict['vertices'] = []
#         l1 = list(Q.exterior.coords)
#         x = set(list(zip(*list(Q.exterior.coords)))[0])
#         y = set(list(zip(*list(Q.exterior.coords)))[1])
        
#         if len(x) >= len(y):
#             for i in range(len(l1) - 1):
#                 x1, y1 = l1[i]
#                 x2, y2 = l1[i + 1]
#                 if y1 == y2:
#                     continue
#                 else:
#                     temp_dict['vertices'].append({"x": x1, "y": y1})
#                     temp_dict['vertices'].append({"x": x2, "y": y2})
                    
#         else:
#             for i in range(len(l1) - 1):
#                 x1, y1 = l1[i]
#                 x2, y2 = l1[i + 1]
#                 if x1 == x2:
#                     continue
#                 else:
#                     temp_dict['vertices'].append({"x": x1, "y": y1})
#                     temp_dict['vertices'].append({"x": x2, "y": y2})
                    
#         plat_dict.append(temp_dict)    
#     return plat_dict

