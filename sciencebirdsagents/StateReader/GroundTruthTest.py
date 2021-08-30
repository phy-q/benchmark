# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:39:32 2020

@author: kimxu
"""
import json

class GroundTruthTest:
    
    def __init__(self,json):
        
        self.json = json
        
    
    def test_ground(self):
        
        # ground object should be the first object, and the value needs to be a float number
        self.ground_level = self.json[0]["yindex"]
        assert isinstance(self.ground_level, float), "ground level yindex {} is not float".format(self.ground_level)
        
        # ground color map is always a empty list
        self.ground_colormap = self.json[0]["colormap"]
        assert self.ground_colormap == [], "ground colormap {} is not an empty list".format(self.ground_colormap)
        
        # ground object has 4 keys, namely id, type, yindex, colormap
        keys_to_check = ["id","type","yindex","colormap"]
        
        for key in keys_to_check:
            assert key in self.json[0].keys(), "gound {} is not in {}".format(key, self.json[0].keys)
        
        assert len(self.json[0].keys()) == 4, "length of ground object {} is not 4".format(len(self.json[0].keys()))

    def test_traj(self):
        
        # the traj object should be the second object
        traj = self.json[1]
        
        # traj object has 4 keys, namely id, type, location, colormap
        keys_to_check = ["id","type","location","colormap"]
        
        for key in keys_to_check:
            assert key in self.json[1].keys(), "traj {} is not in {}".format(key, self.json[0].keys())
        
        assert len(self.json[1].keys()) == 4, "length of ground object {} is not 4".format(len(self.json[0].keys()))
        
        # traj color map is always a empty list
        traj_colormap = traj["colormap"]
        assert traj_colormap == [], "traj colormap {} is not an empty list".format(traj_colormap)
        

    
    def test_slingshot(self):
        
        for j in self.json:
            if j['type'] == "Slingshot":
                slingshot = j
                break
        
        #slingshot has only 4 vertices
        vertices = slingshot['vertices']
        assert len(vertices) == 4, "slingshot has {} vertices: {} ".format(len(vertices), vertices)
        
        #check for general vertices validaty
        self._check_vertices(vertices)
        
        # min_y = 480
        # for v in vertices:
        #     if v['y'] < min_y:
        #         min_y = v['y']
        
        # # the minimum y value of the slingshot should equal to the ground level
        # assert abs(min_y - self.ground_level) < 1e-6, "slingshot y position {} is not close to the ground level {}".format(min_y, self.ground_level)

    def test_other_objects(self):
        
        for j in self.json:
            
            if j['type'] != "Slingshot" and j['type'] != "Ground" and j['type'] != "Trajectory":
                
                # check for the number of keys
                assert len(j) == 4, "{} is not 4 keys".format(j)

                
                # other objects has 4 keys, namely id, type, location, colormap
                keys_to_check = ["id","type","vertices","colormap"]                
                for key in keys_to_check:
                    assert key in j.keys(), "object {} is not in {}".format(key, j.keys())
                    
                # check vertices and colormaps
                vertices = j["vertices"]
                colormap = j["colormap"]
                
                self._check_vertices(vertices)
                self._check_colormap(colormap)
                
                
    def check(self):
        self.test_ground()
        self.test_traj()
        self.test_slingshot()
        self.test_other_objects()
        print("ground truth passed validity check")


        
    def _check_colormap(self,colormap):

        '''
        @input:
            
            colormap: a list of dictionay contains colormaps per object
        
        check if:
        1. colormap is made of keys of "color" and "percent"
        2. the corrsponding color values should between 0 and 255 (integer), percent between 0 and 1 (float)
        3. the sum of percent for for one object should be close to 1
        '''        
        percent_list = []
        
        for pair in colormap:
            
            assert len(pair) == 2, "{} has more than 2 keys".format(pair)
            
            color = pair['color']
            percent = pair['percent']
            
            assert isinstance(color,int), 'color {} is not integer'.format(color)
            assert isinstance(percent,float), 'vertices {} is not float'.format(percent)
        
            assert 0 <= color <= 255, 'color {} is out of normal range'.format(color)
            assert 0 < percent <= 1, 'percent {} is out of normal range'.format(percent)
            
            percent_list.append(percent)
            
        # check if the percent sum to 1
        assert abs(sum(percent_list) - 1) < 1e-6, "percent sum {} not close to one".format(percent_list)
        
        
        
            
    def _check_vertices(self,vertices):
        '''
        @input:
            
            vertices: a list of dictionay contains vertoces points per object
        
        check if:
        1. vertices is made of keys of "x" and "y"
        2. the corrsponding x values should between 0 and 840, y should be less than 480 - y can be negative as suggested by jochen
        3. check if the vertices form a line
        '''
        x_list = []
        y_list = []
        
        for pair in vertices:
            
            assert len(pair) == 2, "{} is not 2 keys".format(pair)
            
            x = pair['x']
            y = pair['y']
            
            assert isinstance(x,float), 'vertices {} is not float'.format(x)
            assert isinstance(y,float), 'vertices {} is not float'.format(y)
        
            assert 0 <= x <= 840, 'vertices x {} is out of normal range'.format(x)
            assert y <= 480, 'vertices y {} is out of normal range'.format(y)
            
            x_list.append(x)
            y_list.append(y)
            
        # check if the vertices form a line
        # -> if they have the same x values or y values
        assert len(set(x_list)) != 1, "vertices {} has same x values".format(x_list)
        assert len(set(y_list)) != 1, "vertices {} has same y values".format(y_list)
        
         

if __name__ == "__main__":
    js = json.load(open("_GTData.json",'r'))
    
    tester = GroundTruthTest(js)
    tester.test_ground()
    tester.test_traj()
    tester.test_slingshot()
    tester.test_other_objects()