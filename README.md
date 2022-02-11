Code repository for Jetson Nano and Realsense  
ssh -Y tony@ip address  
with TX2 to use integrated camera use:  
gst-launch-1.0 nvarguscamerasrc ! nvvidconv ! xvimagesink  
