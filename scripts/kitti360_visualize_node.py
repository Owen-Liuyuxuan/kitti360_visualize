#!/usr/bin/env python3
import numpy as np
import rospy 
import cv2
from utils import kitti360_utils, ros_util
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Int32, Bool, Float32

class Kitti360VisualizeNode:
    """ Main node for data visualization. Core logic lies in publish_callback.
    """
    def __init__(self):
        rospy.init_node("Kitti360VisualizeNode")
        rospy.loginfo("Starting Kitti360VisualizeNode.")

        self.left_image_publish  = rospy.Publisher("/kitti360/left_camera/image", Image, queue_size=1, latch=True)
        self.right_image_publish = rospy.Publisher("/kitti360/right_camera/image", Image, queue_size=1, latch=True)
        self.left_camera_info    = rospy.Publisher("/kitti360/left_camera/camera_info", CameraInfo, queue_size=1, latch=True)
        self.right_camera_info   = rospy.Publisher("/kitti360/right_camera/camera_info", CameraInfo, queue_size=1, latch=True)
        # self.bbox_publish        = rospy.Publisher("/kitti360/bboxes", MarkerArray, queue_size=1, latch=True)
        self.lidar_publisher     = rospy.Publisher("/kitti360/lidar", PointCloud2, queue_size=1, latch=True)
        # self.image_pc_publisher  = rospy.Publisher("/kitti360/left_camera_pc", PointCloud2, queue_size=1, latch=True)
        
        self.KITTI360_raw_dir = rospy.get_param("~KITTI360_RAW_DIR", None)
        self.image_pc_depth  = float(rospy.get_param("~Image_PointCloud_Depth", 5))
        self.update_frequency= float(rospy.get_param("~UPDATE_FREQUENCY", 8))

        self.index = 0
        self.published = False
        self.sequence_index = 0
        self.pause = False
        self.stop = True

        self.meta_dict = kitti360_utils.get_files(self.KITTI360_raw_dir, self.index)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_frequency), self.publish_callback)
        rospy.Subscriber("/kitti360/control/index", Int32, self.index_callback)
        rospy.Subscriber("/kitti360/control/stop", Bool, self.stop_callback)
        rospy.Subscriber("/kitti360/control/pause", Bool, self.pause_callback)


    def stop_callback(self, msg):
        self.published=False
        self.stop = msg.data
        self.sequence_index = 0
        self.publish_callback(None)

    def pause_callback(self, msg):
        self.pause = msg.data      
        self.published=False

    def index_callback(self, msg):
        self.index = msg.data
        self.sequence_index = 0
        self.meta_dict = kitti360_utils.get_files(self.KITTI360_raw_dir, self.index)
        self.publish_callback(None)
        
    def publish_callback(self, event):
        if self.stop: # if stopped, falls back to an empty loop
            return
        
        meta_dict = self.meta_dict
        if meta_dict is None:
            rospy.logwarn("meta_dict from kitti360_utils.get_files is None, current_arguments {}"\
                .format([self.KITTI360_raw_dir, self.index]))
            return

        length = min([len(meta_dict['key_frames'])])
        if length == 0:
            rospy.logwarn("No sequence found at {} index {}".format(self.KITTI360_raw_dir, self.index))
            return
        self.sequence_index = (self.sequence_index) % length

        P0 = meta_dict["calib"]["P0"]
        P1 = meta_dict["calib"]["P1"]
        R0_rect = meta_dict["calib"]["T_rect02cam0"]
        R1_rect = meta_dict["calib"]["T_rect12cam1"]
        T_cam2velo = meta_dict["calib"]["T_cam2velo"]
        T_image0 = meta_dict["calib"]["cam_to_pose"]["T_image0"]
        T_image1 = meta_dict["calib"]["cam_to_pose"]["T_image1"]
        ros_util.publish_transformation(np.linalg.inv(T_cam2velo), 'left_camera', 'lidar')
        ros_util.publish_transformation(T_image0, 'base_link', 'left_camera')
        ros_util.publish_transformation(R0_rect,  'left_camera', 'left_rect')
        ros_util.publish_transformation(T_image1, 'base_link', 'right_camera')
        ros_util.publish_transformation(R1_rect,  'right_camera', 'right_rect')
        ros_util.publish_transformation(meta_dict["poses"][self.sequence_index], "odom", "base_link")

        if self.pause: # if paused, all data will freeze
            return

        
        frame_idx = meta_dict['key_frames'][self.sequence_index]
        if meta_dict['left_image'] is not None:
            left_image = cv2.imread(meta_dict["left_image"][frame_idx])
            ros_util.publish_image(left_image, self.left_image_publish, self.left_camera_info, P0, "left_camera")
        if meta_dict['right_image'] is not None:
            right_image = cv2.imread(meta_dict["right_image"][frame_idx])
            ros_util.publish_image(right_image, self.right_image_publish, self.right_camera_info, P1, "right_camera")

        point_cloud = np.fromfile(meta_dict["lidar"][frame_idx], dtype=np.float32).reshape(-1, 4)
        #point_cloud = point_cloud[point_cloud[:, 0] > np.abs(point_cloud[:, 1]) * 0.2 ]
        ros_util.publish_point_cloud(point_cloud, self.lidar_publisher, "lidar")
        
        self.sequence_index += 1

def main():
    node = Kitti360VisualizeNode()
    rospy.spin()

if __name__ == "__main__":
    main()
