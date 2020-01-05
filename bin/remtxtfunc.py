
from matplotlib import pyplot as plt
import numpy as np

import cv2
#import skimage as sk



'''       
ifile = "D:/IGCAR/test/MTD CWP.jpg"
#ifile = "D:/IGCAR/test/MTD CWP1 A-B.jpg"

ifile = "D:/IGCAR/To Vishal/,IGCAR SS 25NB 17,18 X.jpg" # medval -  51.0   Thresh -  96.0  ret2 -64.0 pixidf 10 block 99
#ifile = "D:/IGCAR/ToClassify/PL 21797 0-1.jpg"  # medval -  36.0   Thresh -  75.0  ret2 -64.0
#ifile = "D:/IGCAR/toupload/images/ultra.png"

ifile = "D:/IGCAR/Rounded/,IGCAR SS 50NB 19 Y.jpg"
'''

def remvtext(ifile,inimg):
    img1 = cv2.imread(ifile)
    imgo = inimg
    imgh,imgw,_ = img1.shape
    
    img = img1[30:imgh-30, 30:imgw-30]

    ret,preimg111 = cv2.threshold(imgo,  127 ,255,cv2.THRESH_OTSU )
    medval = np.median(imgo)

    thresh = (ret * 2)

    if ret > 80:
        thresh = 170

    if ret > 90:
        thresh = 220
    if ret<80:
        thresh = 145
        
    
    # print('ret - ',ret,'med val - ',medval,'thresh - ',thresh)
    img1 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1][:,:,0]
    #img1 = cv2.threshold(imgo, thresh, 255, cv2.THRESH_BINARY)

    #medval = np.median(img1)

    # print('med val - ',medval)

    sigma = 0.33
    canval = thresh
    '''
    if thresh > 150:
        canval = 150
    else:
        canval = thresh
    '''
    low = int(max(0, (1.0 - sigma) * canval))
    high = int(min(255, (1.0 + sigma) * canval))

    # print('Low and high - ',low,high)

    imgcan = cv2.Canny(img1, low, high)
    final = imgo
    contours, hierarchy = cv2.findContours(imgcan.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)

        (x, y, w, h) = cv2.boundingRect(c)
        x1 = x+w
        y1 = y+h

        if area < 30:
            continue
        
        #print('x,y,..',x,x1,y,y1)
        cv2.rectangle(final,(x,y),(x1,y1),(255, 255, 255), -1)
    #dst = cv2.inpaint(img, img1, 150, cv2.INPAINT_NS)

    #wrtstatus = cv2.imwrite('D:/IGCAR/toupload/images/processedimg.jpg', final)
        
    return (final)




