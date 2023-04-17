import cv2

class VideoEditor:
    """class that takes as input a set of numpy frames on 8 bit, and performs drawing on them"""
    def __init__(self,np_frames):
        self.initial_frames = np_frames
        self.np_frames = self.initial_frames.copy()

    def reset_frames(self):
        self.np_frames = self.initial_frames.copy()
        
    def draw(self,key_pts_per_frame,color=(0,255,0)):
        """
            key_pts_per_frame : Iterable of 3D numpy frames (dim0 indexes the detections,
            dim1 indexes the points and dim2 indexes the 2 dimensions of points)
        """
        for keypoints_per_detection,frame in zip(key_pts_per_frame,self.np_frames):
            for keypoints in keypoints_per_detection:
                for pt in keypoints:
                    cv2.circle(frame,pt,1,color)