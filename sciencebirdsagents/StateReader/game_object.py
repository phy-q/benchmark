from StateReader.cv_utils import Rectangle
from enum import Enum

class GameObjectType(Enum):
    UNKNOWN = 'unknown'
    GROUND = 'ground'
    PLATFORM = 'platform'
    SLING = 'slingshot'
    REDBIRD = 'redBird'
    YELLOWBIRD = 'yellowBird'
    BLUEBIRD = 'blueBird'
    BLACKBIRD = 'blackBird'
    WHITEBIRD = 'whiteBird'
    PIG = 'pig'
    ICE = 'ice'
    WOOD = 'wood'
    ROUNDWOOD = 'roundWood'
    STONE = 'stone'
    TERRAIN = 'terrain'
    TNT = 'TNT'

class GameObjectShape(Enum):
   
    RectTiny = 0
    RectSmall = 1
    Rect = 2
    RectMedium = 3
    RectBig = 4
    RectFat = 5
    
    SquareTiny = 6
    SquareSmall = 7
    Square = 8
    SquareBig = 9
    SquareHole = 10
    
    Triangle = 11
    TriangleHole = 12
    
    CircleSmall = 13
    Circle = 14

    
    
class GameObject(Rectangle):
    counter = 0
    def __init__(self, mbr, type, vertices = None, shape = GameObjectShape.Rect, angle = 0):
        super().__init__(rectangle = mbr)
        self.id = GameObject.counter
        GameObject.counter += 1
        #object type
        self.type = type
        self.shape = shape
        self.angle = angle
        self.vertices = vertices

##for test purpose
if __name__ == "__main__":
    print(GameObjectType('pig'))
