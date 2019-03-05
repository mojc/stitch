# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:35:25 2017

@author: Mojca
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

#%%right -> left
def stitch(img1_c,img2_c):
    img1 = cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)
    # Initiate SIFT detector
    stif = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = stif.detectAndCompute(img1,None)
    kp2, des2 = stif.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    good = sorted(matches, key = lambda x:x.distance)[:12]
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # Draw first 10 matches.
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    out = cv2.copyMakeBorder(img2,top=0,bottom=0,left=0,right=img2.shape[1],borderType=cv2.BORDER_CONSTANT)
    dst = cv2.warpPerspective(img1,M,(2*img1.shape[1],img1.shape[0]))
    locsb = np.where(np.logical_and(dst>0 , out > 0)) #both black
    locse = np.where(np.logical_xor(dst>0 , out > 0)) #either black
    #Overlap area
    x0, x1 = min(locsb[1])+50, max(locsb[1])-50
    X=x1-x0
    # Calculate gradient masks
    gradr = np.zeros(out.shape)
    gradr[:,x0:x1]=np.repeat([np.linspace(0,1,X)],gradr.shape[0],axis=0)
    gradr[:,x1:]=1
    gradl = 1-gradr #Left gradient is 1-right gradient
    gradl[locse[0],locse[1]]=1
    gradr[locse[0],locse[1]]=1
    final = gradl*out+gradr*dst #output is  weighted sum of both
    out=final
    cv2.imwrite('output.jpg',out)
    #% back to color
    out = cv2.copyMakeBorder(img2_c,top=0,bottom=0,left=0,right=img2_c.shape[1],borderType=cv2.BORDER_CONSTANT)
    dst = cv2.warpPerspective(img1_c,M,(2*img1_c.shape[1],img1.shape[0]))
    # back to color
    def add_alpha(pic,alpha):    
        b_channel, g_channel, r_channel = cv2.split(pic)    
        return cv2.merge((b_channel*alpha, g_channel*alpha, r_channel*alpha))
    final1,final2= add_alpha(out,gradl),add_alpha(dst,gradr)
    out=final1+final2
    cv2.imwrite('outputc.jpg',out)
    return out
#%% right->left

out = stitch(cv2.imread('dunajM.jpg'), cv2.imread('dunajL.jpg')) #dela
#out = stitch(cv2.imread('dunajR.jpg'), cv2.imread('dunajM.jpg')) #dela - dobro vidna črta na robu
#out = stitch(cv2.imread('gradR.jpg'), cv2.imread('gradM.jpg')) #ne dela, napačne korespondenčne točke
#out = stitch(cv2.imread('gradM.jpg'), cv2.imread('gradL.jpg')) #dela zmazan dimnik
#out = stitch(cv2.imread('hisaR.jpg'), cv2.imread('hisaL.jpg')) #dela
#out = stitch(cv2.imread('obhR.jpg'), cv2.imread('obhM.jpg')) #ne dela
#out = stitch(cv2.imread('obhM.jpg'), cv2.imread('obhL.jpg')) #dela, ghost in malo popačena preslikava


