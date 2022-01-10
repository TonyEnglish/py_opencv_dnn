import cv2
import pyrealsense2 as rs
import numpy as np

HAAR_CASCADE_XML_FILE_FACE = "/home/tony/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml"

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

def faceDetect():
    # Obtain face detection Haar cascade XML files from OpenCV
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_FILE_FACE)

    #configure realsense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    for s in device.sensors:
        print(s.get_info(rs.camera_info.name)) 
        # for 435i should get Stereo Module; RGB Camera; Motion Module
    print(device_product_line) #D400
    pipeline.start(config)
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            cv2.namedWindow("Face Detection Window", cv2.WINDOW_AUTOSIZE)
            if not color_frame or not depth_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)

            # Create rectangle around the face in the image canvas
            for (x_pos, y_pos, width, height) in detected_faces:
                cv2.rectangle(image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 0), 2)
                dist = depth_frame.get_distance(int(x_pos + width/2),int(y_pos + height/2))
                print(dist)

            # Remove background - Set pixels further than clipping_distance to grey
            grey_color = 153
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, image)

            images = np.hstack((image, bg_removed))

            cv2.imshow("Face Detection Window", images)

            key = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if key == 27:
                break

    finally:
        #video_capture.release()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    faceDetect()
