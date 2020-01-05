import cv2
import numpy as np
#import autocanny as ac
from PIL import Image
import os
import bin.penevalfunc as pene
import bin.remtxtfunc as rt

from matplotlib import pyplot as plt




def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def defmarking(mfile,penetype,weldtype):

    maxarea = 19999
    ###print (mfile)
    gray0 = cv2.imread(mfile,cv2.IMREAD_GRAYSCALE)
    #temfile = "D:/IGCAR/template/a.jpg"
    temploc = "bin/template/"
    gray00 = gray0.copy()
    '''
    for file in os.listdir(temploc):
        
        template = cv2.imread((os.path.join(temploc, file)),cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(gray0,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(gray00, pt, (pt[0] + w, pt[1] + h), (0,), -1)
        
    gray2 = gray00.copy()
    '''
    #im = Image.open(mfile)

    #dpi = im.info['dpi']
    #imgdpi = dpi[0]

    #print('DPI - ',imgdpi)
    #new_image = np.zeros(gray.shape, gray.dtype) 

    #gray1 = cv2.bitwise_not(gray)



    imgh,imgw = gray0.shape


    #gray = gray2[30:imgh-30, 30:imgw-30]
    grayori = gray0[30:imgh-30, 30:imgw-30]

    gray10 = rt.remvtext(mfile,grayori.copy())
    gray = gray10.copy()
    
    penedia=0
    penedia = pene.penevalcm(grayori,10)
    penediamm = 1.27   #(2.54 * 0.16)
    if penedia ==0:
      penedia = 30

    
    if penetype=='Penetrometer 5' or penetype == 'Penetrometer 7' or penetype=='Penetrometer 10':
        penediamm = 1.02
    elif penetype=='Penetrometer 12':
        penediamm = 1.27
    elif penetype=='Penetrometer 15':
        penediamm = 1.52        
    elif penetype=='Penetrometer 17':
        penediamm = 1.78
    elif penetype=='Penetrometer 20':
        penediamm = 2.03       
    elif penetype=='Penetrometer 25':
        penediamm = 2.54       
    elif penetype=='Penetrometer 30':
        penediamm = 3.05       
    elif penetype=='Penetrometer 35':
        penediamm = 3.56
    elif penetype=='Penetrometer 40':
        penediamm = 4.06       
    elif penetype=='Penetrometer 45':
        penediamm = 4.57
    else:
        penediamm = 5.08
    

    # print('penetype - ',penetype,'  penedia cm - ',penediamm)

    imgh,imgw = gray.shape
    ###print (imgh,imgw)


    ret0,preimg111 = cv2.threshold(gray,  127 ,255,cv2.THRESH_OTSU )

    medval = np.median(gray)

    ###print ('medval - ',medval,'  Thresh - ',ret0)
    if medval > 100:    # to be updated
        maxarea = 29999


    madj = 450

    st1x = 0
    st1y = 0

    en1x = int(imgw/2) - madj
    en1y = int(imgh/2) - madj

    st2x = int(imgw/2) + madj
    st2y = 0

    en2x = imgw
    en2y = int(imgh/2) - madj

    st3x = 0
    st3y = int(imgh/2) + madj

    en3x = int(imgw/2) - madj
    en3y = imgh 

    st4x = int(imgw/2) + madj
    st4y = int(imgh/2) + madj

    en4x = imgw
    en4y = imgh 


    #final = gray.copy()
    grayproc = gray.copy()
    cv2.rectangle(grayproc,(st1x,st1y),(en1x,en1y),(0),cv2.FILLED)
    cv2.rectangle(grayproc,(st2x,st2y),(en2x,en2y),(0),cv2.FILLED)
    cv2.rectangle(grayproc,(st3x,st3y),(en3x,en3y),(0),cv2.FILLED)
    cv2.rectangle(grayproc,(st4x,st4y),(en4x,en4y),(0),cv2.FILLED)

    graygamma1 = adjust_gamma(grayproc,gamma=0.5)
    graygamma = adjust_gamma(graygamma1,gamma=1.85)

    clahe = cv2.createCLAHE(clipLimit= 0.75, tileGridSize=(8,8))
    graynew = clahe.apply(graygamma)


    stval = 250

    if imgh > imgw:
        xstval = int(imgw/6)
        ystval = int(imgh/11)
        xstart = xstval
        xend = (imgw-xstval)
        ystart = ystval
        yend = (imgh - ystval) 
    else:
        xstval = int(imgw/11)
        ystval = int(imgh/6)
        xstart = xstval
        xend = (imgw-xstval)
        ystart = ystval
        yend = (imgh - ystval)
        
    ###print ('start vale - ',xstart,xend,ystart,yend)

        

    if medval < 40:
        pixdif = 7
    elif medval < 45:
        pixdif = 4
    elif medval < 55:
        pixdif = 10
    elif medval < 80:    
        pixdif = 3
    else:
        pixdif = 7


    ###print('Max Area - ',maxarea)

    if medval < 40:
        pixdif = 7
    elif medval > 100:
        pixdif = 7

    else:
        pixdif = 5

    pixdif = 14


    maskimg = cv2.fastNlMeansDenoising(graynew,None,30,7,21)

    #maskimgeq = cv2.equalizeHist(maskimg)
    #maskimgeq1 = cv2.fastNlMeansDenoising(maskimgeq,None,30,7,21)

    #maskimgeq2 = adjust_gamma(maskimg,gamma=0.6)

    #maskimgeq3 = cv2.convertScaleAbs(maskimgeq2,alpha=10.0/maskimgeq2.max(), beta=0.)
    conmaxval = False
    
    if weldtype == 'normal' or weldtype == 'Normal':
        tungsten = False
    else:
        tungsten = True
        
    
    #tungsten = False
    #if (medval==76 and ret0 == 111) or (medval==45 and ret0 == 70) :
    if tungsten == True:
        output = cv2.bitwise_not(maskimg)  #maskimg1
        pixdif = 8
        conmaxval = True
    else:
        output = maskimg
        
    preimg = output

    k = np.ones((5,5),np.uint8)

    preimgd = cv2.dilate(preimg, k, iterations=2)
    preimge = cv2.erode(preimgd, k, iterations=2)


    preimgbl = cv2.GaussianBlur(preimge,(5,5),0)
    const = 4
    '''
    if medval > 130:
        blocksz = 199
    elif medval < 37:
        blocksz = 291
    else:
        blocksz = 99

    if medval == 37:    
        blocksz = 199
        
    if medval==131 and ret0 == 110:
        const = 1
        blocksz = 47
        pixdif = 17
    '''
    blocksz = (int(medval/2)*2) + 1
    #blocksz = 99
    if medval > 85 and medval < 100:
        blocksz = 71
        pixdif = 14
        #const = 2
    
    if medval > 85 and medval < 100 and medval > ret0:
        if ret0 < 55:
            blocksz = 199
            pixdif = 20
            const = 4
            maxarea = 49999

    if medval > 89 and medval < 100 and ret0 > 59 and ret0 < 70:
        blocksz = 99
        pixdif = 14
        const = 4
        maxarea = 15999
        
    if medval > 79 and medval < 85 and medval > ret0:
        blocksz = 127
        pixdif = 14
        const = 4
        maxarea = 19999

    '''
    if medval > 99 and medval < 120:
        #blocksz = 71
        pixdif = 14
        const = 3
        maxarea = 39999
    '''

    if medval == 37 or medval == 36 or medval == 115:    
        blocksz = 199

    if medval==131 and ret0 == 110:
        const = 1
        blocksz = 73
        pixdif = 23

    '''
    if medval==55 and ret0 == 141:
        const = 1
        blocksz = 199
        #pixdif = 3
    '''
    if medval == 53 or medval == 122:
        blocksz = 99

        
        
    if medval < 50:
        blocksz = 99
        const = 4
        pixdif = 11
        
    if medval==35 and ret0 == 139:
        const = 0
        blocksz = 199
        pixdif = 7

    if medval < 25:   # to be updated
        blocksz = 99
        pixdif = 14

    if ret0 < 35 and medval < 35 :
        blocksz = 89
        const = 2
        pixdif = 12

    if ret0 > 130:
        #const = 3
        pixdif = 10

    if ret0 > 145:
        const = 4
        pixdif = 14

    if medval > 120:
        maxarea = 39999
        const = 3
        pixdif = 14

    if medval > 145:
        maxarea = 39999
        #const = 3
        #pixdif = 10


    if medval < 52 and ret0 > 100:
        blocksz = 249
        const = 4
        pixdif = 14

    if medval < 25 and ret0 > 100:
        blocksz = 199
        const = 3
        pixdif = 14

    if medval > 110 and ret0 > 110:
        blocksz = 199
        const = 1
        pixdif = 7
    


    ###print ('pixdif - ',pixdif)


    ###print('block - ',blocksz,const)

    preimgath = cv2.adaptiveThreshold(preimgbl,127,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blocksz,const)



    preimg1 = cv2.fastNlMeansDenoising(preimgath,None,30,7,21)


    premedval = np.median(preimg1)

    ret2,preimg111 = cv2.threshold(preimg1,  127 ,255,cv2.THRESH_OTSU )

    ###print ('pre med - ',premedval,ret2)
    sigma = 0.33

    if premedval > 40:
        low = int(max(0, (1.0 - sigma) * premedval))
        high = int(min(255, (1.0 + sigma) * premedval))
    else:
        low = int(max(0, (1.0 - sigma) * ret2))
        high = int(min(255, (1.0 + sigma) * ret2))
        
    #preimgce = cv2.Canny(preimg5, (edgval * .67 ), (edgval * 1.33))
    preimgce = cv2.Canny(preimg1, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #preimgopen = cv2.morphologyEx(preimgce, cv2.MORPH_OPEN, kernel)

    preimgclos = cv2.morphologyEx(preimgce, cv2.MORPH_CLOSE, kernel)

    #preimgclos = preimgce

    ###print ('edge val - ',low,high)


    #new, contours, hierarchy = cv2.findContours(preimgclos.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(preimgclos.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)

    final = cv2.cvtColor(grayori.copy(),cv2.COLOR_GRAY2RGB)
    final1 = final.copy()
    #contours = sorted(contours, key=cv2.contourArea)
    ###print (len(contours))
    ###print (contours)
    mcount = 0
    mc = 0
    arrcnt = 0
    linearcnt = 0
    roundcnt = 0

    xlist = []
    ylist = []
    rtnlist =[]    
    # loop over the digit area candidates
    for c in contours:
        cnt = contours[mcount]
        
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        hullarea = cv2.contourArea(hull)       
        
        '''
        hull = cv2.convexHull(c)
        hullarea = cv2.contourArea(hull)   
        minrectarea = cv2.minAreaRect(c)
        if hullarea > 0:
            solidity = area/hullarea
        else:
            solidity = 0   
        '''
        (x, y, w, h) = cv2.boundingRect(c)
        calarea = (w * h)
        calareaper = (calarea * 0.15)
        '''
        if cv2.isContourConvex(cnt) == True:
            closcontour = 'True'
        else:
            closcontour = 'False'
        '''

        mcount = mcount + 1

        #cv2.drawContours(final, [cnt], 0, (255,255,0),2)
        '''
        if calareaper > area:
            if calareaper < 99 or calareaper > 29999: # or peri > area:
                continue
            if peri > calareaper:
               continue
        else:
        '''   
        if area < 19 or area > maxarea: # or peri > area:
            continue
        
        
        if (peri*1.1) > area:
            #if peri< 100:
            continue

        if area < 9000:
            if (area*1.75) < hullarea:
                # print ('x, y, w, h  -- ', x, y, w, h)
                # print('Area - ',area,hullarea)
                continue
     
        
        graymask = grayori.copy() # gray.copy()

        mask = np.zeros(graymask.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        newimg = graymask[y:y+h,x:x+w]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(newimg)

        #conintensity = cv2.mean(graymask,mask=mask)
        #print ('min and max  - ',min_val, max_val, min_loc, max_loc)
        #plt.subplot(121),plt.imshow(mask ,cmap = 'gray')
        #plt.show()


        M = cv2.moments(c)  
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #print ('center - ',cX,cY)
        '''
        if cX >= imgw:
            cX = imgw-1

        if cY >= imgh:
            cY = imgh-1
        '''
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        #print (extLeft)
        leftx = extLeft[0]
        lefty = extLeft[1]
        #print (leftx,lefty)
        rightx = extRight[0]
        righty = extRight[1]

        topx = extTop[0]
        topy = extTop[1]

        botx = extBot[0]
        boty = extBot[1]
        #leftx = leftx - 1
        #print ('Left x - ',leftx,lefty)

        '''
        if leftx >= imgh:
            leftx = imgh-1

        if lefty >= imgw:
            lefty = imgw-1

        if rightx >= imgh:
            rightx = imgh-1

        if righty >= imgw:
            righty = imgw-1
        '''

        if (rightx - leftx) < (boty - topy):
            centerx = int(((topx + botx)/2))
            centery = int(topy + ((boty - topy)/2))
        else:    
            centerx = int(leftx + ((rightx - leftx)/2))
            centery = int(lefty + ((righty - lefty)/2))

        #print('X,y - ',centerx,centery,cX,cY)
        cpixel1 = int(final1[[centery],[centerx],0])
        cpixel2 = int(final1[[cY],[cX],0])

        if cpixel1 < cpixel2:
            cpixel = cpixel1
        else:
            cpixel = cpixel2
        if conmaxval == True:
            cpixel = max_val
        else:
            cpixel = min_val
            
        #centery = righty
        '''
        ###print('Center 1 - ',centerx,centery)
        
        centerx = cX
        centery = cY
       ###print('Center 2 - ',centerx,centery)
        '''
        
        if (rightx - leftx) < 10:   # 10
            continue
        if (boty - topy) < 10:   # 10
            continue
        
        '''
        if solidity > 0.15:
            if (peri*1.1) > area:
                continue
        else:
            if (leftx == topx or lefty == topy or rightx == botx or righty == boty):
                continue
        '''
        #if (rightx - leftx) < 100 or (boty - topy) < 100 :   # 10
        
        
        #print ('center - ',leftx,rightx,centerx)
        #cpixel = int(final1[cY,cX,0])
        #cpixel = int(final1[[centery],[centerx],0])
        #cpixel = int((min_val+max_val)/2)
        #cpixel = min_val
        #lpixel = int(final1[[lefty],[leftx-1],0])
        if leftx > 1:
            lpixel = int((int(final1[[lefty],[leftx],0]) + int(final1[[lefty],[leftx-1],0]) +int(final1[[lefty],[leftx-2],0]))/3)
        else:
            lpixel = int(final1[[lefty],[leftx],0])
            
        if rightx > imgw-1:
           rpixel = int((int(final1[[righty],[rightx],0]) + int(final1[[righty],[rightx+1],0]) +int(final1[[righty],[rightx+2],0]))/3)
        else:
           rpixel = (int(final1[[righty],[rightx],0]) )
           
                   
        if leftx > (imgw/2):
            lpixel = rpixel

        #print('Pix value - ',cpixel,min_val,max_val)
        #cpixel,c2,c3 = final1.getpixel((centerx,centery))
        #lpixel,l2.l3 = final1.getpixel((leftx,lefty))

     

        #print ('abs value - ',cX,cY,leftx,lefty,lpixel, cpixel, abs(lpixel - cpixel))
        
        if abs(lpixel - cpixel) < pixdif:    #11  6
            continue
        
        x1 = x+w
        y1 = y+h

        if ((x > xstart and x1 < xend) and (y > ystart and y1 < yend)):
        #if 1==1:
            #mmarea = areaincm(area)
            mmarea = 0
            if penedia == 0:
                penemmarea = 0
            else:
                
                penemmarea = ((penediamm / penedia) * area)

            if w > h:
                #mmlength = areaincm(w)
                mmlength = 0
                if penedia > 0:
                    penemmlength = ((penediamm / penedia) * w)
                else:
                     penemmlength = 0   
            else:
                #mmlength = areaincm(h)
                mmlength = 0
                if penedia > 0:
                    penemmlength = ((penediamm / penedia) * h)
                else:
                    penemmlength = 0
            
                        
            xlist.append([])            
            ylist.append([])            
            xlist[arrcnt].append(x)
            xlist[arrcnt].append(penemmlength)
            ylist[arrcnt].append(y)
            ylist[arrcnt].append(penemmlength)

            if h > w:
                dcnt = int(h/w)
            else:
                dcnt = int(w/h)
            if dcnt >= 3:
                linearcnt = linearcnt + 1
                xlist[arrcnt].append('L')
                ylist[arrcnt].append('L')
            else:
                roundcnt = roundcnt + 1
                xlist[arrcnt].append('R')
                ylist[arrcnt].append('R')
            
            arrcnt = arrcnt + 1
                    
           ###print ('length of c: ', len(c))    
           ###print (c[0],c[len(c)-1])
           ###print ('  ')
            #print('minrectarea - ',minrectarea)

           ###print ('Length - ',len(c))
           ###print ('x, y, w, h  -- ', x, y, w, h)
                
            #print (leftx,lefty,rightx,righty)
            #print (topx,topy,botx,boty)
            #print ('Contour intensity - ',conintensity)
           ###print ('peri - ',peri, area)
           ###print ('defect area in centimeters(DPI) - ',mmarea,' in centimeters(penetro) - ',penemmarea)

           ###print ('defect length (DPI) - ',mmlength, '  penetro length - ',penemmlength)
            #print ('pixel value = ',cpixel)
           ###print('Pixel value - ',cpixel1,cpixel2,cpixel)
            
           ###print ('min val  ',min_val,max_val, min_loc)
           ###print (extLeft,extRight,extTop,extBot)
                
           ###print (centerx,centery,extLeft)
           ###print ('left pixel value = ',lpixel)
           ###print (' ')
            
            #cv2.rectangle(final,(x,y),(x1,y1),(255, 255, 0), 2)

            cv2.drawContours(final, [cnt], 0, (0,255,255),2)

            #cv2.drawContours(final, c, -1, (255,255,0),2)
            '''
            plt.subplot(121),plt.imshow(preimg,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(final,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            #plt.suptitle(meth)

            plt.show()
            '''

    xlist.sort(key=lambda x: x[0])
    ylist.sort(key=lambda x: x[0])
    

    arrlen = len(xlist)
    if arrlen > 0:

        xdiff = xlist[(arrlen-1)][0] - xlist[0][0]

        ydiff = ylist[(arrlen-1)][0] - ylist[0][0]

        if ydiff > (xdiff*2):
            prntext = '    (From top to bottom)'
            rtnlist = ylist
        else:
            prntext = '    (From left to right)'
            rtnlist = xlist


    return (grayori,final,prntext,rtnlist,linearcnt,roundcnt)



'''  
plt.subplot(121),plt.imshow(preimgath,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(final,cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#plt.suptitle(meth)

plt.show()

'''
