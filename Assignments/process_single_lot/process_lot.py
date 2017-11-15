import numpy as np
import cv2 as cv
from scipy.spatial import distance

from matplotlib import pyplot as plt
from json import dumps,loads
import pprint as pp
from sys import argv, exit


def load_spaces(definition_file):
    f = open(definition_file,'r')
    spaces = loads(f.read())

    return spaces

def extract_points(space):
    points = []
    for i in range(4):
        x = int(space['contour']['point'][i]['x'])
        y = int(space['contour']['point'][i]['y'])
        points.append((x,y))
    return points

def draw_parking_space(points,img):
    colors = [(0,0,255),(0,255,0),(255,0,0),(255,255,0)]
    for i in range(4):
        x1 = points[i][0]
        y1 = points[i][1]
        x2 = points[(i+1)%4][0]
        y2 = points[(i+1)%4][1]
        cv.line(img, (x1,y1),(x2,y2),colors[i], 2)

def make_parallelogram(p,type=0):
    """
    Types: 0 = smallest area , 1 = largest area , 2 = avg area
    """
    for i in range(4):
        a = p[i]
        b = p[(i+1) % 4]
        dst = distance.euclidean(a,b)
        print(dst)

def new_space_image(pixels,width,height):
    blank_image = np.zeros((height,width,3), np.uint8)


if __name__=='__main__':

   # if len(argv) < 3:
    #    exit()

    #definition_file = '/Users/nikhil/jsonfile.json'
    #image_file = '/Users/nikhil/imagefile.jpg'

    s=0
    
    spaces = load_spaces('/Users/nikhil/jsonfile.json')
    img = cv.imread('/Users/nikhil/imagefile.png')
    
    img=np.asarray(img)
    #cv.imshow('img',img)
    #cv.waitKey(0)
    
    for space in spaces:
        points = extract_points(space)
        xmin=points[3][0] 
        xmax=points[1][0] 
        ymin= points[0][1] 
        ymax= points[2][1] 
        print('nextone')
        print('printing cropped image')
        #cv.show("spaces/" + str(s) + ".jpg" , crop_img)
        s = s+1
        make_parallelogram(points)
        draw_parking_space(points,img)
        crop=img[ymin:ymax,xmin:xmax]
        cv.imwrite("spaces/"+ str(s) +".png", crop)
        hsv = cv.cvtColor(crop,cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        np.savetxt('histograms/'+str(s)+".csv", hist, delimiter=',')
    # Typical opencv methods to show images
    cv.imshow('Draw01',img)
    cv.waitKey(0)
    # plt.imshow(img)
    # plt.show()
