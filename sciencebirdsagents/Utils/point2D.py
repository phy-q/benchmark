from math import hypot

class Point2D:
    '''point on a 2d Euclidean plane'''

    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __str__(self):
        return "Point(%s,%s)"%(self.X, self.Y)

    def distance(self, p):
        dx = self.X - p.X
        dy = self.Y - p.Y
        return hypot(dx, dy)

    def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.X == other.X and self.Y == other.Y 
            else:
                return False
