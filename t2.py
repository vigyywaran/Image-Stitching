# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

def Matching_Keypoints(imgs,i_img,central_img,descs,kps):
    sift1 = cv2.xfeatures2d.SIFT_create()
    kp1,desc1 = sift1.detectAndCompute(central_img,None)
    match={}
    for i in range(0, len(kps[i_img])):
        score=[]
        pos=0
        min_score = 99999
        for j in range(0,len(kp1)):
            diff = descs[i_img][i] - desc1[j]
            score.append(np.sqrt(np.sum(np.square(diff))))
            if (score[j] < min_score):
                min_score = score[j]
                pos = j
                #match_seq[kps[img_1][i]]=kps[img_2][pos]
        if (min_score < 100):
            match[kp1[pos]]=kps[i_img][i]
    return match

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    overlap_arr = np.zeros((len(imgs),len(imgs)))
    sift = cv2.xfeatures2d.SIFT_create()
    kps = []
    descs = []
    matching={}
    matching_seq = {}
    #print (len(imgs))
    for i in range(0,len(imgs)):
        #imgs[i]=cv2.cvtColor(imgs[i],cv2.CV_8U)
        #imgs[i]=cv2.cvtColor(imgs[i],COLOR_BGR2GRAY)
        kp,desc = sift.detectAndCompute(imgs[i],None)
        #print (imgs[i].shape)
        kps.append(kp)
        descs.append(desc) 
    #print (len(kps), len(kps[0]))
    #print (len(descs), len(descs[0]), len(descs[0][0]))
    #print (kps[0][0])
    ### MATCHING ####
    for img_1 in range(0,len(imgs)):
        for img_2 in range(0,len(imgs)):
            #print (img_1, img_2)
            if (img_1==img_2):
                overlap_arr[img_1][img_2]=1
            else:
                match = {}
                match_seq={}
            #fil_name = 'Match_'+str(img_1)+'_'+str(img_2)+'.txt'
            #fil1=open(fil_name,'w')
                for i in range (0, len(kps[img_1])):
                    score=[]
                    pos=0
                    min_score = 99999
                    for j in range(0,len(kps[img_2])):
                        diff = descs[img_1][i] - descs[img_2][j]
                        score.append(np.sqrt(np.sum(np.square(diff))))
                        if (score[j] < min_score):
                            min_score = score[j]
                            pos = j
                    #match_seq[kps[img_1][i]]=kps[img_2][pos]
                    if (min_score < 100):
                        match[kps[img_1][i]]=kps[img_2][pos]
                if (len(match.keys())>=int(0.2*len(kps[img_1])) or len(match.keys())>=int (0.2*len(kps[img_2]))):    
                    overlap_arr[img_1][img_2]=1
                    matching[(img_1,img_2)] = match
                    matching_seq[(img_1,img_2)]=min((len(match.keys())/len(kps[img_1])),(len(match.keys())/len(kps[img_2])))
                else:
                    overlap_arr[img_1][img_2]=0
    #print (overlap_arr
    #print (matching_seq)
    #flag=0
    central_flag=0
    for i in range(0,len(imgs)):
        if ((np.sum(overlap_arr[i])==len(imgs)) and (central_flag==0)):
            ## imgs[i] -> Centeral image <i is connected to every other image so central image>
            central_img = i
            central_flag=1
            break
    #print (central_img)
    #print (matching.keys())
    #prev_results = np.zeros(((imgs[central_img].shape[0]),(imgs[central_img].shape[1]),3))
    h,w,c = imgs[central_img].shape
    prev_results = imgs[central_img]
    #print (results[0].shape)
    #print (results.shape)
    h_prev,w_prev=(0,0)
    for i in range(0,len(imgs)):
        #print (i, central_img)
        if (central_img==i):
            continue
        else:
            h1,w1,c1 = imgs[i].shape
            h2,w2,c2  = prev_results.shape
            if (i==0):
                src = np.float32([kp_src.pt for kp_src in matching[(central_img,i)].keys()]).reshape(-1,1,2)
                dest=np.float32([kp_dest.pt for kp_dest in matching[(central_img,i)].values()]).reshape(-1,1,2)
            else:
                #cv2.imwrite('prev_result.jpg',prev_results)
                kp1,desc1=sift.detectAndCompute(prev_results,None)
                match_kp = Matching_Keypoints(imgs,i,prev_results,descs,kps)
                src = np.float32([kp_src.pt for kp_src in match_kp.keys()]).reshape(-1,1,2)
                dest=np.float32([kp_dest.pt for kp_dest in match_kp.values()]).reshape(-1,1,2)
            M,mask = cv2.findHomography(src,dest,cv2.RANSAC,5) 
            pt1=np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
            pt2=np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
            pt2_ = cv2.perspectiveTransform(pt2, M)
            pts = np.concatenate((pt1, pt2_), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin,-ymin]
            Mt = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
            warp_result = cv2.warpPerspective(prev_results, Mt.dot(M), (xmax-xmin, ymax-ymin))
            warp_result[t[1]:h1+t[1],t[0]:w1+t[0]] = imgs[i]
            h_warp, w_warp,c_warp = warp_result.shape
            results=warp_result[0:h_warp,0:w_warp,0:3]
            prev_results = np.zeros((int(results.shape[0]*1),int(results.shape[1]*1),3))
            h,w,c = results.shape
            prev_results = results[0:h,0:w,:]
            prev_results = cv2.normalize(prev_results, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite('result_t2.png',prev_results)
    print ("Done")
    return overlap_arr 
if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
