from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

semper0_pic = cv.imread('Resources/semper0.jpeg')
semper0_pic = cv.cvtColor(semper0_pic,cv.COLOR_BGR2GRAY)

semper1_pic = cv.imread('Resources/semper1.jpeg')
semper1_pic = cv.cvtColor(semper1_pic,cv.COLOR_BGR2GRAY)


def get_point_data(img_name):
    img = cv.imread('Resources/' + img_name + '.jpeg')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, desc = sift.detectAndCompute(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite(img_name + '_kp.jpg',img)
    return (kp, desc)


def get_matches(points1, points2):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(points1, points2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)
    return good_matches

def draw_lines():
    img = Image.open("Resources/semper0.jpeg")
    draw = ImageDraw.Draw(img)
    for index in range(len(list_kp1)):
        #extremely long matches are usually errors, remove
        srcx, srcy = list_kp1[index]
        destx,desty = list_kp2[index]
        if np.sqrt(np.power(srcx - destx, 2) + np.power(srcy - desty, 2)) > .2 * img.width:
            continue
        draw.ellipse((srcx-3, srcy-3, srcx+3, srcy+3))
        draw.line([(srcx, srcy), (destx, desty)], fill='red', width=1)
        
    img.show()


semper0_kp, semper0_desc = get_point_data('semper0')
semper1_kp, semper1_desc = get_point_data('semper1')


matches = get_matches(semper0_kp, semper1_kp)
list_kp1 = [semper0_kp[mat.queryIdx].pt for mat in matches]
list_kp2 = [semper1_kp[mat.trainIdx].pt for mat in matches]


draw_lines()

