from typing import Any, List
from PIL import Image
import numpy as np

import time #test time with start = time.time(); end = time.time(); print(end-start)

class Circle:
    def __init__(self, cx, cy, cr):
        self.cx = cx
        self.cy = cy 
        self.cr = cr

    def contains(self, px, py):
        left = pow(self.cx - px, 2) + pow(self.cy-py, 2)
        right = pow(self.cr, 2)
        return left < right

    def invert(self, px:float, py:float):
        pt = [0,0]
        r2 = pow(self.cr, 2)
        dx = px - self.cx
        dy = py - self.cy
        dist2 = (dx * dx) + (dy * dy)
        pt[0] = ((r2 * dx) / dist2) + self.cx
        pt[1] = ((r2 * dy) / dist2) + self.cy
        #print(pt)
        return np.array(pt)

#line defined by 2 points
class PlaneLine:
    """plane line figure to generate contains along with the circles

    A line defined by 2 points (A and B)
    some theta can be used instead where point A is on some horizontal line
    and said line with vector AB have degree theta 

    this makes it easier to initialize an actual horzontal line
    where you specify the x axis with A and set theta to 0

    aboveLine defines the inside of the "line" circle being one side of the line. 
    used in contains and invert methods
    """

    def __init__(self, A:List[float]=[0,0], B:List[float]=None, 
                        theta:float=None,
                        radians:bool=True,
                        aboveLine:bool=True ) -> None:
        self.A:np.ndarray[Any] = np.array(A)
        self.aboveLine = aboveLine
        if theta is not None:
            if radians == True:
                self.theta = theta
            else:
                self.theta = np.radians(theta)
            self.B:np.ndarray[Any] = np.array([np.cos(theta), np.sin(theta)]) #test #gets 
        elif B is not None:
            self.B:np.ndarray[Any] = np.array(B)
            

    def contains(self, px:float, py:float) -> bool:
        #AB = np.cross(self.A,self.B)
        #AP = np.cross(self.A, np.array([px,py]))
        AB = self.B - self.A
        AP = np.array([px,py]) - self.A
        side:int = 1
        if self.aboveLine == False:
            side *= -1
        
        result = np.cross(AB, AP) * side
        if result >= 0: #error here using 3rd method?
            return True
        else: #technically: 0 means it's on the line which will be counted
            return False

    def invert(self, px:float, py:float) -> np.ndarray[Any]:
        #reflection over line 
        P = np.array([px,py])
        AB = self.B - self.A
        AP = P - self.A
        num = AB.dot(AP)
        den = AB.dot(AB)
        proj = (num / den) * AB
        orthogonalToAB = (P - proj) 
        return P - (2*orthogonalToAB)

class Coloring:
    scalar = float
    def __init__(self) -> None:
        pass
    
    def __hsv_to_rgb(self, h:scalar, s:scalar, v:scalar ) -> tuple:
        """take hue, saturation, value (lightness) to RGB

        input: takes in float between 0-1 for every input 
        h: 0-360 degrees -> 0-100 percent of that -> 0-1 (float)
        s: 0-100% -> 0-1 (float)
        v: 0-100% -> 0-1 (float)
        output:
        R,G,B as 0-1 (float) 
        returns  
        """
        if s:
            if h == 1.0: h = 0.0
            i = int(h*6.0); f = h*6.0 - i
            
            w = v * (1.0 - s)
            q = v * (1.0 - s * f)
            t = v * (1.0 - s * (1.0 - f))
            
            #note: make %255 remainder instead?
            if i==0: return (v, t, w)
            if i==1: return (q, v, w)
            if i==2: return (w, v, t)
            if i==3: return (w, q, v)
            if i==4: return (t, w, v)
            if i==5: return (v, w, q)
        else: return (v, v, v)

    
    def hsv_to_rgb(self, h:scalar, s:scalar, v:scalar ) -> tuple:
        """returns 255, 255, 255 based RGB color
        using hidden method 
        """
        rgb = self.__hsv_to_rgb(h,s,v)
        return (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)

    def getHue(self, iterations: int, maxIterations:int, loops:int) -> float:
        """0-1 decimal representing looping degree of hue from a given iteration"""
        return (((iterations + 1) * loops) % maxIterations) / maxIterations

    #hsb (is hue sat bright | hsv: hue sat val)
    #use this method for most calls!
    def getRGB(self, iterations:int, maxIterations:int, loops:int = 2) -> tuple:
        """returns (255,255,255) RGB tuple 

        based on percentage of iterations compared to maxIterations
        """
        hue = self.getHue(iterations, maxIterations, loops)
        sat = 1
        val = 1
        return self.hsv_to_rgb(hue, sat, val)

#ended up not using this function, not 100% sure it works
def getOrthogonalPts(c1:List[float], r1:float, c2:List[float], r2:float) -> np.ndarray[Any]:
    y:float
    ax = c1[0]; ay = c1[1]
    bx = c2[0]; by = c2[1]
    denom = 2 * (ax - bx)
    const1 = ((ax*ax) + (ay*ay)) - ((bx*bx) - (by*by)) / denom 
    scalar1 = (2 * (ay - by)) / denom
    #x = c - sy
    #x^2 - 2ax(x) + (ax)^2 
    a = (scalar1 * scalar1) + 1
    b = (-2 * scalar1) + (2 * ax * scalar1) - (2 * ay)
    c = (ax * ax) + (ay * ay) + (const1 * const1) - (2*ax*const1) - (r2 * r2)
    descriminant = (b * b) - (4 * a * c)
    y1 = (-b + np.sqrt(descriminant))/(2*a)
    y2 = (-b - np.sqrt(descriminant))/(2*a)
    x1 = c - (scalar1 * y1)
    x2 = c - (scalar1 * y2)
    return np.array([[x1,y1], [x2,y2]])

def circumcenter(x1, y1, x2, y2, x3, y3):
    """
    Calculate the circumcenter of a triangle given three points.

    Args:
        x1, y1, x2, y2, x3, y3 (float): Coordinates of the three points.

    Returns:
        tuple: Coordinates (x, y) of the circumcenter.
    """
    # Calculate the circumcenter using the formula
    D = 2 * (x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2)
    Ox = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    Oy = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D

    r = np.sqrt((x1 - Ox)**2 + (y1 - Oy)**2)

    return Ox, Oy, r

def iniCircles():
    circles = [
        PlaneLine([0,0], theta=0, aboveLine=False),
        Circle(cx=1, cy=1, cr=1),
        Circle(cx=-1, cy=1, cr=1),
        Circle(cx=0, cy=0.25, cr=0.25),
    ]   
    return circles

#needs to be global
circles = iniCircles()

def getInversionCount(pointX, pointY, maxIterations):
    X = 0
    Y = 1
    count = 0
    pt:np.ndarray = np.array([pointX, pointY])
    for i in range(maxIterations):
        inverted = False
        for c in circles:    
            if c.contains(pt[0], pt[1]):
                inverted = True
                pt = c.invert(pt[0], pt[1])
                count += 1
                break
        if inverted == False or np.isnan(pt.any()):
            return count 
    return count 

def calcScale(oMin, oMax, sMin, sMax):
    """sets screen scale to xy plane scale"""
    dO = oMax - oMin
    dS = sMax - sMin
    return dS/dO

def main(): 
    #window sizing
    maxIterations = 100
    SIZE = 600
    min = -1
    max = 1
    xMin = min
    xMax = max 
    yMin = min
    yMax = max
    #window constant scaling variables
    xScale = calcScale(0,SIZE-1,xMin, xMax)
    x_int = xMin
    yScale = calcScale(0,SIZE-1,yMin, yMax)
    y_int = yMin

    #create speedup lists
    color_loop = 2 #how many times to loop around 360 deg angle for hue (decided by inversionCount)
    color = Coloring()
    #creates association btw a count (index) and an RGB tuple
    possible_rgb_values = [color.getRGB(i, maxIterations, color_loop) for i in range((maxIterations*color_loop) + 1)]
    rgb_pixels = np.empty((SIZE, SIZE, 3), dtype=np.uint8)

    start = time.time()
    for row in range(SIZE):
        for col in range(SIZE):
            x = (col * xScale) + x_int
            y = (row * yScale) + y_int
            count = getInversionCount(x,y, maxIterations)

            if count > 0:
                #rgb_pixels[row][col] = color.getRGB(count, maxIterations)
                rgb_pixels[row][col] = possible_rgb_values[count] 
            else:
                rgb_pixels[row][col] = (0,0,0)
    
    end = time.time()
    print("Image Generation Time: ", end-start)

    img = Image.fromarray(rgb_pixels)
    img.save('image.png')
    #img.show()

#helper
def generateBlackPicture(SIZE:float) -> np.ndarray:
    rgb_pixels = np.full((SIZE, SIZE, 3), fill_value=0, dtype=np.uint8)
    return rgb_pixels

#testing
def main2():    
    #rgb = colorsys.hsv(20, 1, 1)
    color = Coloring()
    rgb = color.hsv_to_rgb(359/360.0, 1.0, 1.0)
    print(rgb)
    print(pow(2,4))
    for i in range(101):
        print("count:", i)
        #rgb = getRGB(iterations=i, maxIterations=100)
        print("hue", color.getHue(iterations=i, maxIterations=100))

#test plane line object
def main3():
    line = PlaneLine([0,0], theta=0)
    print(line.B)
    print(line.contains(px=1, py=300)) #contains works
    print(line.invert(px=1, py=1)) #works?

#test properties of pixel grid
def main4():
    rgbColors = generateBlackPicture(SIZE=800)
    row = 0
    col = 1
    rgbColors[row][col] = (255,255,255)
    img = Image.fromarray(rgbColors)
    img.save('image.png')

if __name__=="__main__": 
    main()
     
