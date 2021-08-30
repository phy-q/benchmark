import torch
class NDSparseMatrix:
    def __init__(self,c,h,w):
        self.elements = {}
        self.h = h
        self.w = w
        self.c = c

    def addValue(self, c,y,x, value):
        self.elements[(c,y,x)] = value

    def toTensor(self):
        ret = torch.zeros((self.c,self.h,self.w))
        for key, value in self.elements.items():
            c,y,x = key
            ret[c,y,x] = value
        return ret
