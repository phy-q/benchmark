#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 09:58:27 2019

@author: chengxue
"""
import sys

sys.path.append('..')
sys.path.append('./src')
import numpy as np
# import os
from StateReader.game_object import GameObject, GameObjectType
from StateReader.cv_utils import Rectangle
from Utils.NDSparseMatrix import NDSparseMatrix


class NotVaildStateError(Exception):
    """NotVaildStateError exceptions"""
    pass


class SymbolicStateDevReader:
    def __init__(self, json, *args, **kwargs):

        '''
        json : a list of json objects. the first element is int id, 2nd is png sreenshot
        if sreenshot is required, and the rest of them is the ground truth of game
        objects

        look_up_matrix: matrix of size n * 256, where n is the number of tempelet we used, 256 represents the 8bit color value

        look_up_obj_type: length n array, storing the type corrsponding to the look_up_matrix

        '''

        self.type_transformer = {
            'bird_blue': 'blueBird',
            'bird_yellow': 'yellowBird',
            'bird_black': 'blackBird',
            'bird_red': 'redBird',
            'bird_white': 'whiteBird',
            'platform': 'platform',
            'pig_basic_big': 'pig',
            'pig_basic_small': 'pig',
            'pig_basic_medium': 'pig',
            'TNT': 'TNT',
            'Slingshot': 'slingshot',
            'ice': 'ice',
            'stone': 'stone',
            'wood': 'wood',
            'unknown': 'unknown'
        }

        self.alljson = []
        json = json[0]['features']
        for j in json:
            if j['properties']['label'] != 'Platform':
                self.alljson.append(j)

        self._parseJsonToGameObject()

    def get_symbolic_image_sparse(self, h: int, w: int) -> np.array:
        '''
        get_symbolic_image returns a hxwx12 numpy array as to represent the game state.
        channel object
        1. slingshot
        2. red bird
        3. yellow bird
        4. blue bird
        5. white bird
        6. black bird
        7. pigs
        8. wood objects
        9. ice objects
        10. stone objects
        11. tnts
        12. platforms
        Objects are represented as 1 in the channel in the h,w block
        '''
        ret = NDSparseMatrix(c=12, w=w, h=h)
        x_size = 640
        y_size = 480
        x_range = np.linspace(0, x_size-1, w)
        y_range = np.linspace(0, y_size-1, h)
        channel_idx = {
            'blueBird': 3, 'yellowBird': 2, 'blackBird': 5, 'redBird': 1, 'whiteBird': 4, 'platform': 11, 'pig': 6,
            'TNT': 10, 'slingshot': 0, 'ice': 8, 'stone': 9, 'wood': 7}

        for obj_type in self.allObj:
            c = channel_idx[obj_type]
            for obj in self.allObj[obj_type]:
                top_left_x, top_left_y = obj.top_left
                bottom_right_x, bottom_right_y = obj.bottom_right

                # allocate to the slot
                for i in range(len(x_range) - 1):
                    if x_range[i] < top_left_x <= x_range[i + 1]:
                        top_left_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < top_left_y <= y_range[i + 1]:
                        top_left_slot_y = i
                for i in range(len(x_range) - 1):
                    if x_range[i] < bottom_right_x <= x_range[i + 1]:
                        bottom_right_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < bottom_right_y <= y_range[i + 1]:
                        bottom_right_slot_y = i

                for x in range(top_left_slot_x, bottom_right_slot_x + 1):
                    for y in range(top_left_slot_y, bottom_right_slot_y + 1):
                        ret.addValue(c=c, x=x, y=y, value=1)

        return ret

    def get_symbolic_image(self, h: int, w: int) -> np.array:
        '''
        get_symbolic_image returns a hxwx12 numpy array as to represent the game state.
        channel object
        1. slingshot
        2. red bird
        3. yellow bird
        4. blue bird
        5. white bird
        6. black bird
        7. pigs
        8. wood objects
        9. ice objects
        10. stone objects
        11. tnts
        12. platforms
        Objects are represented as 1 in the channel in the h,w block
        '''
        ret = np.zeros((12, h, w), dtype=np.float)
        x_size = 640
        y_size = 480
        x_range = np.linspace(0, x_size, w)
        y_range = np.linspace(0, y_size, h)
        channel_idx = {
            'blueBird': 3, 'yellowBird': 2, 'blackBird': 5, 'redBird': 1, 'whiteBird': 4, 'platform': 11, 'pig': 6,
            'TNT': 10, 'slingshot': 0, 'ice': 8, 'stone': 9, 'wood': 7}

        for obj_type in self.allObj:
            c = channel_idx[obj_type]
            for obj in self.allObj[obj_type]:
                top_left_x, top_left_y = obj.top_left
                bottom_right_x, bottom_right_y = obj.bottom_right

                # allocate to the slot
                for i in range(len(x_range) - 1):
                    if x_range[i] < top_left_x <= x_range[i + 1]:
                        top_left_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < top_left_y <= y_range[i + 1]:
                        top_left_slot_y = i
                for i in range(len(x_range) - 1):
                    if x_range[i] < bottom_right_x <= x_range[i + 1]:
                        bottom_right_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < bottom_right_y <= y_range[i + 1]:
                        bottom_right_slot_y = i

                for x in range(top_left_slot_x, bottom_right_slot_x + 1):
                    for y in range(top_left_slot_y, bottom_right_slot_y + 1):
                        ret[c, y, x] = 1

        return ret

    def is_vaild(self):
        '''
        check if the stats received are vaild or not

        for vaild state, there has to be at least one pig and one bird.
        '''

        pigs = self.find_pigs()
        birds = self.find_birds()

        if pigs and birds:
            return True
        else:
            return False

    def _parseJsonToGameObject(self):
        '''
        convert json objects to game objects
        '''

        self.allObj = {}
        # find the type of all object

        # 1. vectorize the dictionary of colors
        obj_num = 0
        obj_total_num = len(self.alljson)
        obj_types = np.zeros(obj_total_num).astype(str)
        self.obj_ids = {}

        for j in self.alljson:
            obj_type_splited = j['properties']['label'].split("_")
            if len(obj_type_splited) == 1:
                type = j['properties']['label']
            elif obj_type_splited[0] == 'bird':
                type = "_".join(obj_type_splited[:2])
            elif obj_type_splited[0] == 'pig':
                type = "_".join(obj_type_splited[:3])
            else:
                type = obj_type_splited[0]

            obj_types[obj_num] = type
            obj_num += 1

        obj_num = 0
        for j in self.alljson:

            if j['properties']['label'] == "Slingshot":

                rect = self._getRect(j)
                contours = j['geometry']['coordinates']
                vertices = contours[0]

                game_object = GameObject(rect, GameObjectType(self.type_transformer["Slingshot"]), vertices)

                try:
                    self.allObj[self.type_transformer["Slingshot"]].append(game_object)
                except:
                    self.allObj[self.type_transformer["Slingshot"]] = [game_object]

                self.obj_ids[j['properties']['id']] = game_object

            elif j['properties']['label'] == "Ground" or j['properties']['label'] == "Trajectory":
                pass

            else:
                rect = self._getRect(j)
                contours = j['geometry']['coordinates']
                vertices = contours[0]
                game_object = GameObject(rect, GameObjectType(self.type_transformer[obj_types[obj_num]]), vertices)

                try:
                    self.allObj[self.type_transformer[obj_types[obj_num]]].append(game_object)
                except:
                    self.allObj[self.type_transformer[obj_types[obj_num]]] = [game_object]

                self.obj_ids[j['properties']['id']] = game_object
            obj_num += 1

    def get_symbolic_image_flat(self, h, w):
        ret = np.zeros((h, w), dtype=np.float)
        x_size = 640
        y_size = 480
        x_range = np.linspace(0, x_size-1, w)
        y_range = np.linspace(0, y_size-1, h)
        channel_idx = {
            'blueBird': 3, 'yellowBird': 2, 'blackBird': 5, 'redBird': 1, 'whiteBird': 4, 'platform': 11, 'pig': 6,
            'TNT': 10, 'slingshot': 0, 'ice': 8, 'stone': 9, 'wood': 7}

        for obj_type in self.allObj:
            c = channel_idx[obj_type]
            for obj in self.allObj[obj_type]:
                top_left_x, top_left_y = obj.top_left
                bottom_right_x, bottom_right_y = obj.bottom_right

                # allocate to the slot
                for i in range(len(x_range) - 1):
                    if x_range[i] < top_left_x <= x_range[i + 1]:
                        top_left_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < top_left_y <= y_range[i + 1]:
                        top_left_slot_y = i
                for i in range(len(x_range) - 1):
                    if x_range[i] < bottom_right_x <= x_range[i + 1]:
                        bottom_right_slot_x = i
                for i in range(len(y_range) - 1):
                    if y_range[i] < bottom_right_y <= y_range[i + 1]:
                        bottom_right_slot_y = i

                for x in range(top_left_slot_x, bottom_right_slot_x + 1):
                    for y in range(top_left_slot_y, bottom_right_slot_y + 1):
                        if ret[y, x] == 0:
                            ret[y, x] = c

        return ret, self.obj_ids

    def _getRect(self, j):
        '''
        input: json object
        output: rectangle of the object
        '''
        contours = j['geometry']['coordinates']
        vertices = contours[0]

        x = []
        y = []
        for v in vertices:
            x.append(int(float(v[0])))
            y.append(int(float(v[1])))
        points = (np.array(y), np.array(x))
        rect = Rectangle(points)
        return rect

    def find_bird_on_sling(self, birds, sling):
        sling_top_left = sling.top_left[1]
        distance = {}
        for bird_type in birds:
            if len(birds[bird_type]) > 0:
                for bird in birds[bird_type]:
                    # print(bird)
                    distance[bird] = abs(bird.top_left[1] \
                                         - sling_top_left)
        min_distance = 1000
        for bird in distance:
            if distance[bird] < min_distance:
                ret = bird
                min_distance = distance[bird]
        return ret

    def find_pigs(self):
        ret = self.allObj.get('pig', None)
        return ret

    def find_platform(self):
        ret = self.allObj.get('Platform', None)
        return ret

    def find_slingshot(self):
        ret = self.allObj.get('slingshot', None)
        return ret

    def find_birds(self):
        ret = {}
        for key in self.allObj:
            if 'Bird' in key:
                ret[key] = self.allObj[key]
        if len(ret) == 0:
            return None
        else:
            return ret

    def find_blocks(self):
        ret = {}
        for key in self.allObj:
            if 'wood' in key or 'ice' in key or 'stone' in key or 'TNT' in key:
                ret[key] = self.allObj[key]
        if len(ret) == 0:
            return None
        else:
            return ret
