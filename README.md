# Elder
Midterm Project Proposal: Object Recognition Lock by Machine Vision Methods Kai Yu (78762168), Ciel Xiong(95903774)

May 2017

project description and features

The goal of this project is to achieve machine vision on the PYNQ board, namely edge detection, as well as pattern detection, etc. We can use this technology practically, such as using in a camera to detect some identical feature of objects. Then use the object's feature to build a lock for important data access on PYNQ board by applying Machine Vision Methods. The PYNQ Board then can be used as a safe storage tool with a unique lock of some unique object, such as a real key or a pen, as long as the object can be placed stably on a table plate. The user need the objects set as the key to unlock data on board by taking a photo of the object by camera connecting to the board.

The expectation of the program can use web cam to capture the instantaneous image then apply pattern recognition. After a user take a photo of their object, the PYNQ board will measure brightness and color on each pixel and then convert the image to grey level image, and compute Square Gradient Magnitude (SGM), and apply Hough transform to detect identities such as lines and circles on the image. If the image matches the key stored on Board, then the users are able to get access to important data on Board.

plan week 1: Purchase Web Camera for testing on Board, and implement a basic program to use a Button on PYNQ board to take pictures by the camera, and test the functionality and compatibility of Board and Camera; week 2: Initialize HDMI /IO, make sure we can stream, control, and process images through HDMI. Start to write pattern recognition code by python. week 3: Write algorithms of different machine vision, such as line detection, object recognition. Complete the needed codes and test them on different general images. Start write code of Lock and Key part. week 4: Apply our algorithm into the board, make sure that the board can give us a correct output of machine vision. Try to implement the program on images camera takes, and generate correct output. Finish the Lock and Key program. week 5: Test, and try to figure out more useful machine vision algorithms, also try to use different programming language such as C code and assembly language instead of python. Continue debugging and try to add some more features on board to improve functionality.

progress Research on edge detection method, Hough transform, and Square Gradient Magnitude. Test image processing examples on PYNQ Board.

sensors/actuators use

HDMI cable,Web camera (Tentative: Logitech HD Webcam C310);

tasks of each member Kai Yu: Prepare the PYNQ board, HDMI cable, figure out machine vision algorithm, test on board;

Ciel Xiong: Testing and debug this project, and figure out how to connect web camera with Board. Implement Key and Lock program.
