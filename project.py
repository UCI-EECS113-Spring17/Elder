
# coding: utf-8

# # Final Project
# 
# In this notebook, several filters will be applied to webcam images.
# 
# Those input sources and applied filters will then be displayed either directly in the notebook or on HDMI output.
# 
# To run all cells in this notebook a webcam and HDMI output monitor are required.  
# 

# ## 1. Start HDMI output 
# ### Step 1: Load the overlay

# In[1]:

from pynq import Overlay
Overlay("base.bit").download()


# ### Step 2: Initialize HDMI I/O

# In[2]:

from pynq.drivers.video import HDMI
hdmi_out = HDMI('out')
hdmi_out.start()


# ## 2. Applying OpenCV filters on Webcam input
# ### Step 1: Initialize Webcam and set HDMI Out resolution

# In[3]:

# monitor configuration: 640*480 @ 60Hz
hdmi_out.mode(HDMI.VMODE_640x480)
hdmi_out.start()
# monitor (output) frame buffer size
frame_out_w = 1920
frame_out_h = 1080
# camera (input) configuration
frame_in_w = 640
frame_in_h = 480


# ### Step 2: Initialize camera from OpenCV

# In[4]:

from pynq.drivers.video import Frame
import cv2
from time import sleep
from pynq.board import LED
from pynq.board import RGBLED
from pynq.board import Button
from pynq.board import Switch

leds = [LED(index) for index in range(4)]
rgbleds = [RGBLED(index) for index in [4,5]] 
btns = [Button(index) for index in range(4)]
switches = [Switch(index) for index in range(2)]    


while (1):
    x = 0;
    for btn in btns:
        if (btn.read()==1):
            leds[0].on()
            videoIn = cv2.VideoCapture(0)
            print("Image captured!")
            x = 1;
            break
    if(x==1):
        break
            


videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(videoIn.isOpened()))


# ### Step 3: Send webcam input to HDMI output

# In[5]:

import numpy as np

ret, frame_vga = videoIn.read()

if (ret):
    frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)
    frame_1080p[0:480,0:640,:] = frame_vga[0:480,0:640,:]
    hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8)))
else:
    raise RuntimeError("Error while reading from camera.")


# ### Step 4: Edge detection 
# Detecting edges on webcam input and display on HDMI out.

# In[6]:

import time
frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)

num_frames = 20
readError = 0

start = time.time()
for i in range (num_frames):   
    # read next image
    ret, frame_vga = videoIn.read()
    if (ret):
        laplacian_frame = cv2.Laplacian(frame_vga, cv2.CV_8U)
        # copy to frame buffer / show on monitor reorder RGB (HDMI = GBR)
        frame_1080p[0:480,0:640,[0,1,2]] = laplacian_frame[0:480,0:640,[1,0,2]]
        hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8)))
    else:
        readError += 1
end = time.time()

print("Frames per second: " + str((num_frames-readError) / (end - start)))
print("Number of read errors: " + str(readError))


# ### Step 5: Canny edge detection
# Detecting edges on webcam input and display on HDMI out.
# 
# Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure to be non-edges, so discarded. Those who lie between these two thresholds are classified edges or non-edges based on their connectivity. If they are connected to “sure-edge” pixels, they are considered to be part of edges. Otherwise, they are also discarded. 

# In[7]:

frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)

num_frames = 20

start = time.time()
for i in range (num_frames):
    # read next image
    ret, frame_webcam = videoIn.read()
    if (ret):
        frame_canny = cv2.Canny(frame_webcam,100,110)
        frame_1080p[0:480,0:640,0] = frame_canny[0:480,0:640]
        frame_1080p[0:480,0:640,1] = frame_canny[0:480,0:640]
        frame_1080p[0:480,0:640,2] = frame_canny[0:480,0:640]
        # copy to frame buffer / show on monitor
        hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8)))
    else:
        readError += 1
end = time.time()

print("Frames per second: " + str((num_frames-readError) / (end - start)))
print("Number of read errors: " + str(readError))


# ### Step 6: Show edge image results
# Now use matplotlib to show filtered webcam input inside notebook

# In[8]:

get_ipython().magic('matplotlib inline')


from matplotlib import pyplot as plt
import numpy as np
import os
from pynq.iop.iop_const import PMODA
from pynq.iop import Pmod_OLED
Locked = 0
key=0

Oled=Pmod_OLED(1, "Comparing")



plt.figure(1, figsize=(10, 10))
frame_vga = np.zeros((480,640,3)).astype(np.uint8)
frame_vga[0:480,0:640,0] = frame_canny[0:480,0:640]
frame_vga[0:480,0:640,1] = frame_canny[0:480,0:640]
frame_vga[0:480,0:640,2] = frame_canny[0:480,0:640]

if(switches[0].read()==1):
    cv2.imwrite('lock.jpg',frame_vga)
    print("Lock is set")
    Oled.write("Lock is set!", 0, 200 )
    os.chmod('lockExample.png', 000)
    
    Locked = 1
else:
    cv2.imwrite('key.jpg',frame_vga)
    print("Key is set")
    key=1;

plt.imshow(frame_vga[:,:,[2,1,0]])
plt.show()


# ### Step 7: Apply Hough Transform To Binary Image
# compute Angles and position of lines on Image
# 

# In[9]:








img = frame_vga

edges = frame_canny




lines = cv2.HoughLines(edges,1,np.pi/180,150)
count =0

Oled.clear()
for i in range(len(lines)):
    
    for rho,theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        slope = (y1-y2)/(x1-x2)
        b=y1-slope*x1
        Yaxis=b  ##x=0
        Xaxis=(100-b)/slope  ##y=100
        Xtop=(255-b)/slope ##y=255
        Ytop=slope*255+b ##x=255
        tempY=slope*255
        tempX=255/slope
        
        px1=0
        px2=255
        py1=0
        py2=255
        
        if(slope>=0):
            px1=0
            py1=0
            if((tempY<=255) and (tempY>=0)):
                px2=255
                py2=int(tempY)
            elif((tempX>=0) and (tempY<=255)):
                px2=int(tempX)
                py2=255
        else:
            px1=0
            py1=255
            if((tempX<=255) and (tempX>=0)):
                px2=int(tempX)
                py2=0
            elif(((255+tempY)<=255) and ((255+tempY)>=0)):
                px2=255
                py2=255+int(tempY)
            
        
        '''
        if((Yaxis<=255) and (Yaxis>=100)):
            py1=int(Yaxis)
            px1=0
        elif((Xaxis<=255) and (Xaxis>=0)):
            py1=100
            px1=int(Xaxis)
        
        if((Xtop<=255) and (Xtop>=0)):
            py2=255
            px2=int(Xtop)
        elif((Ytop<=255) and (Ytop>=100)):
            px2=255
            py2=int(Ytop)
        '''
            
        if(count<10):
            Oled.draw_line(px1,py1, px2, py2)
            count=count+1

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
        

plt.imshow(img[:,:,[2,1,0]])
plt.show()


# ### Step 8: Compare Key and Lock
# Blue Rgb for match success, Red for match Fail
# 

# In[10]:





if(key == 1):
    match=0
    LockImg = cv2.imread('lock.jpg')
    LockGray = cv2.cvtColor(LockImg,cv2.COLOR_BGR2GRAY)
    LockEdges = cv2.Canny(LockGray,50,150,apertureSize = 3)
    LockLines = cv2.HoughLines(LockEdges,1,np.pi/180,150)
    
    KeyImg = cv2.imread('key.jpg')
    KeyGray = cv2.cvtColor(KeyImg,cv2.COLOR_BGR2GRAY)
    KeyEdges = cv2.Canny(KeyGray,50,150,apertureSize = 3)
    KeyLines = cv2.HoughLines(KeyEdges,1,np.pi/180,150)
    
    
    
    if(len(LockLines)>len(KeyLines)):
        length = len(KeyLines)
        
    else:
        length = len(LockLines)
        
    
        
    print("Length=" + str(length))
    
    if(length>10):
        length=10
        print("Comparing First 10 lines...")
    
    
    
    LockRho=[]
    LockTheta=[]
    
    
    for i in range(len(LockLines)):
        for rho,theta in LockLines[i]:
            LockRho.append(rho)
            LockTheta.append(theta)
            print("LockRho: " + str(LockRho[i]) +"LockTheta: " + str(LockTheta[i]))
            
    
    KeyRho = []
    KeyTheta = []
    
    for i in range(len(KeyLines)):
        for rho,theta in KeyLines[i]:
            KeyRho.append(rho)
            KeyTheta.append(theta)
            print("KeyRho: " + str(KeyRho[i]) +"KeyTheta: " + str(KeyTheta[i]))
       
            
    for i in range(length):
        if((abs(KeyRho[i]-LockRho[i]<20)) and (abs(KeyTheta[i]-LockTheta[i]<25))):
            match = match + 1
            
    
    
    
    
    if((length==10 and match > 6) or (length < 10 and (length-match < 2))):
        print("Key Match, File Unlock! Match rate = " + str(match))
        rgbleds[0].write(2)
        Oled.write("Key Matched!", 0, 200 )
        os.chmod('lockExample.png', 777)
    
    else:
        print("Match Failed, match rate = " + str(match))
        Oled.write("Match Failed!", 0, 200 )
        rgbleds[0].write(4)
        
    
            

    


# ### Step 7: Release camera and HDMI

# In[11]:

videoIn.release()
hdmi_out.stop()

del hdmi_out


# ###### 
