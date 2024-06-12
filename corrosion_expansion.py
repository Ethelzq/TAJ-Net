import cv2
import numpy as np

def function(th_box):

    kernel= np.ones((5,5))
    erode = cv2.erode(th_box,kernel,iterations=5)
    #cv2.imwrite("result.jpg",erode)
    # cv2.waitKey(0)

    dilate = cv2.dilate(erode,kernel,iterations=5)
    return dilate
    # cv2.imwrite("result2.jpg",dilate)