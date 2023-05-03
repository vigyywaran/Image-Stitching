import torch
import kornia
import torchvision

img1 = torchvision.io.read_image('./images/t1_1.png')
img1=img1.float().unsqueeze(0)
print (img1.shape)
img2 = torchvision.io.read_image('./images/t1_2.png')
#print (img2.shape)
sift = kornia.feature.SIFTFeature(100, upright=False, rootsift=True, device=torch.device('cpu'))
laf,val,desc = sift.forward(img1)
# print (laf.shape)
#print (val.shape)
#print (desc.shape)