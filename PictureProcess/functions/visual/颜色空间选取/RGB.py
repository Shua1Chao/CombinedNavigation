import cv2
import matplotlib.pyplot as plt
import math


img = cv2.imread("../1.jpg")
r,g,b = cv2.split(img)
imgs = [r,g,b,img]
titles = ["r","g","b","srcImage"]
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
h,s,v = cv2.split(img1)
cv2.imshow("Y",h)
cv2.imshow("Cr",s)
cv2.imshow("Cb",v)
cv2.imshow("srcImage",img1)
cv2.waitKey(0)
#imgs = [h,s,v,img1]
#titles = ["h","s","v","srcImage"]
#for i in range(4):
    #plt.subplot(2,2,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    #plt.imshow(imgs[i])
    #plt.title(titles[i])
#plt.show()