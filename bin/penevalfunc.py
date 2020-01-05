import cv2
import os
import numpy as np
from PIL import Image

# import gaborfilter as gf
# import autocanny as ac
from skimage import feature

from matplotlib import pyplot as plt


stx = 0
sty = 0
enx = 0
eny = 0


def penedetect(inimage):
    global stx
    global sty
    global enx
    global eny

    ingray = inimage
    temploc = "bin/penetempold/"
    ingray00 = ingray.copy()

    found = 0
    tempcnt = 0

    for n in range(100, 79, -1):

        for file in os.listdir(temploc):

            template = cv2.imread(
                (os.path.join(temploc, file)), cv2.IMREAD_GRAYSCALE)
            # print(file)
            w, h = template.shape[::-1]
            res = cv2.matchTemplate(ingray, template, cv2.TM_CCOEFF_NORMED)
            if len(res) == 0:
                continue
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            #print('Max Loc - ',max_loc,max_val)
            threshold = (n/100)

            #print('Thresh - ',threshold)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(ingray00, pt, (pt[0] + w, pt[1] + h), (255,), -1)
                stx = pt[0]
                sty = pt[1]
                enx = pt[0] + w
                eny = pt[1] + h
                #print('tmplate - ',file)
                # print('stx...eny',stx,sty,enx,eny)
                #print('Rect Points - ',pt,pt[0],pt[1],w,h)
                found = 1
                # print(file)
                break
            if found == 1:
                break

        if found == 1:
            break
    return ingray00


#oriimg = cv2.imread(mfile,cv2.IMREAD_GRAYSCALE)


#outimg = penedetect(oriimg)


def penevalcm(srcimage, defareapix):

    oriimg = srcimage.copy()
    outimg = penedetect(srcimage.copy())

    outimg[outimg != 255] = 0

    #peneimg = cv2.bitwise_and(oriimg, outimg, mask=None)
    peneimg = cv2.bitwise_and(oriimg, outimg)

    peneimgeq = cv2.equalizeHist(peneimg)

    peneimgeqd = cv2.fastNlMeansDenoising(peneimgeq, None, 30, 7, 21)

    peneimgth = cv2.adaptiveThreshold(
        peneimgeqd, 127, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 5)

    peneimgce = cv2.Canny(peneimgeqd, 50, 150)

    output = peneimgeq.copy()

    # print('stx...eny',stx,sty,enx,eny)

    contours, hierarchy = cv2.findContours(
        peneimgth.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    #print('Len contour - ',len(contours))
    maxradius = 0
    for i in range(len(contours)):
        #color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        if int(radius[i]) < 10 or int(radius[i]) > 100:
            continue

        color = (0, 255, 0)

        if int(centers[i][0]) < stx or int(centers[i][0]) > enx:
            continue
        if int(centers[i][1]) < sty or int(centers[i][1]) > eny:
            continue
        #print((int(centers[i][0]), int(centers[i][1])), int(radius[i]))

        #cv2.drawContours(drawing, contours_poly, i, color)
        # cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            # (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv2.circle(output, (int(centers[i][0]), int(
            centers[i][1])), int(radius[i]), color, 2)
        if int(radius[i]) > maxradius:
            maxradius = int(radius[i])
    '''

    #c = max(contours, key = cv2.contourArea)
    #cv2.drawContours(output, c, -1, 255, 3)

    for c in contours:

         area = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c)



        if area < 10: # or peri > area:
            continue
        print('area - ',area, w)
        print(x, y, w, h)
        cv2.drawContours(output, c, 0, 255, 3)
        print('drawn')
        #break
    '''
    ###print('Max Radius - ',maxradius)
    if maxradius == 0:
        defareacm = 0
    else:
        penediacm = (2.54/.04)
        defareacm = ((penediacm / maxradius) * defareapix)

    # return defareacm
    return (maxradius * 2)
