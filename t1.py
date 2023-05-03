#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades


from re import sub
import cv2
from cv2 import BORDER_CONSTANT
from cv2 import WARP_INVERSE_MAP
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    sift = cv2.xfeatures2d.SIFT_create()
    # SIFT feature extraction for image 1
    kp1,desc1 = sift.detectAndCompute(img1,None)
    #print (np.float32([kp_src.pt for kp_src in kp1]))
    #print (kp1)
    # SIFT feature extraction for image 2
    kp2,desc2 = sift.detectAndCompute(img2,None)
    #print (kp2)
    # Mapping key points with descriptors
    matching = {}
    #file1=open('Matching.txt','w')
    # Map each descriptor in image 1 to all the descriptors in image 2 and
    #compute the score. 
    # Take the least score from that and match the respective key points
    # Iterate over all the keypoints in Image 1 
    for i in range (0, len(kp1)):
        score=[]
        #c=0
        pos=0
        min_score = 99999
        for j in range(0,len(kp2)):
            diff = desc1[i] - desc2[j]
            score.append(np.sqrt(np.sum(np.square(diff))))
            if (score[j] < min_score):
                min_score = score[j]
                pos = j
        if (min_score < 200):
            matching[kp1[i]]=kp2[pos]
    #print (matching)
    src = np.float32([kp_src.pt for kp_src in matching.keys()]).reshape(-1,1,2)
    #print  (src)
    dest=np.float32([kp_dest.pt for kp_dest in matching.values()]).reshape(-1,1,2)
    ## Computing Homography
    M,mask = cv2.findHomography(src,dest,cv2.RANSAC,5)
    print(M)
    h1, w1, c1  = img1.shape
    h2,w2, c2 = img2.shape
    img_warp = np.array([round(w1*1.5),max(h1,h2)])
    warp_result =cv2.warpPerspective(img1,M,img_warp,img2)
    #cv2.imwrite('Warp.jpg',warp_result)
    h_warp,w_warp,c_warp = warp_result.shape
    #print (h_warp,w_warp)
    #print (img2.shape)
    subtract_img_warp = img2 [:,0:w_warp,:] - warp_result [:,0:w_warp,:]
    #print (subtract_img_warp.shape)
    result_1 = np.zeros((h2,w2,3))
    for i in range (0,len(subtract_img_warp)):
        for j in range(0,len(subtract_img_warp[i])):
            for k in range(0,3):
                if ((subtract_img_warp[i,j,k])==img2[i,j,k]):
                    result_1 [i,j,k] = img2[i,j,k]
                elif ((subtract_img_warp[i,j,k]+10)<=img2[i,j,k]):
                    result_1 [i,j,k] = img2[i,j,k]
                else :
                    result_1 [i,j,k] = warp_result[i,j,k]
    for i in range (0,h2):
        for j in range(len(subtract_img_warp[i]),w2):
            for k in range (0,3):
                result_1[i,j,k] = img2[i,j,k]
    cv2.imwrite('result_t1.png',result_1)
    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

