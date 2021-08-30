#!/usr/bin/env python
import os
import numpy as np
import rospy
from visualization_msgs.msg import Marker
import scipy.io as sio
from constants import KITTI_COLORS, KITTI_NAMES

def color_pointcloud(pts, image, T, P2):
    hfiller = np.expand_dims(np.ones(pts.shape[0]), axis=1)
    pts_hT = np.hstack((pts, hfiller)).T #(4, #pts)
    if T.shape == (3, 4):
        T1 = np.eye(4)
        T1[0: 3] = T.copy()
    else:
        T1 = T.copy()
    pts_cam_T = T1.dot(pts_hT) # (4, #pts)

    pixels_T = P2.dot(pts_cam_T) #(3, #pts)
    pixels = pixels_T.T
    pixels[:, 0] /= pixels[:, 2] + 1e-6
    pixels[:, 1] /= pixels[:, 2] + 1e-6
    w_coordinate = pixels[:, 0].astype(np.int32)
    h_coordinate = pixels[:, 1].astype(np.int32)
    w_coordinate[w_coordinate <= 0] = 0
    w_coordinate[w_coordinate >= image.shape[1]] = image.shape[1] - 1
    h_coordinate[h_coordinate <= 0] = 0
    h_coordinate[h_coordinate >= image.shape[0]] = image.shape[0] - 1
    
    bgr = image[h_coordinate, w_coordinate, :]/ 256.0
    return np.concatenate([pts, bgr], axis=1).astype(np.float32)

def read_labels(file):
    """Read objects 3D bounding boxes from a label file.

    Args:
        file: string of path to the label file

    Returns:
        objects: List[Dict];
        object['whl'] = [w, h, l]
        object['xyz'] = [x, y, z] # center point location in center camera coordinate
        object['theta']: float
        object['score']: float
        object['type_name']: string
    """
    objects = []
    with open(file, 'r') as f:
        for line in f.readlines():
            objdata = line.split()
            class_name = objdata[0]
            if class_name in KITTI_NAMES:
                whl = [float(objdata[9]), float(objdata[8]), float(objdata[10])]
                xyz = [float(objdata[11]), float(objdata[12]) - 0.5 * whl[1], float(objdata[13])]
                theta = float(objdata[14])
                if len(objdata) > 15:
                    score = float(objdata[15])
                else:
                    score = 1.0
                objects.append(
                    dict(whl=whl, xyz=xyz, theta=theta, type_name=class_name, score=score)
                )
    return objects

def read_T_from_sequence(file):
    """ read T from a sequence file calib_cam_to_velo.txt
    """
    with open(file, 'r') as f:
        line = f.readlines()[0]
        data = line.strip().split(" ")
        T = np.array([float(x) for x in data[0:12]]).reshape([3, 4])

    T_velo2cam = np.eye(4)
    T_velo2cam[0:3, :] = T
    return T_velo2cam

def read_P01_from_sequence(file):
    """ read P0 and P1 from a sequence file perspective.txt
    """
    P0 = None
    P1 = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("P_rect_00"):
                data = line.strip().split(" ")
                P0 = np.array([float(x) for x in data[1:13]])
                P0 = np.reshape(P0, [3, 4])
            if line.startswith("P_rect_01"):
                data = line.strip().split(" ")
                P1 = np.array([float(x) for x in data[1:13]])
                P1 = np.reshape(P1, [3, 4])
    assert P0 is not None, "can not find P0 in file {}".format(file)
    assert P1 is not None, "can not find P1 in file {}".format(file)
    return P0, P1

def read_extrinsic_from_sequence(file):

    T0 = np.eye(4)
    T1 = np.eye(4)
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("image_00"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T0[0:3, :] = np.reshape(T, [3, 4])
            if line.startswith("image_01"):
                data = line.strip().split(" ")
                T = np.array([float(x) for x in data[1:13]])
                T1[0:3, :] = np.reshape(T, [3, 4])
    

    return dict(
        T_image0=T0, T_image1=T1
    )

def read_poses_file(file):
    key_frames = []
    poses = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            key_frames.append(int(data[0]))
            pose = np.eye(4)
            pose[0:3, :] = np.array([float(x) for x in data[1:13]]).reshape([3, 4])
            poses.append(pose)
    poses = np.array(poses)
    return key_frames, poses

def determine_date_index(pose_dir, index):
    date_times = os.listdir(pose_dir)
    date_times.sort()
    data_time_name = date_times[index]
    rospy.loginfo("Selected date time {}, total_number of date times {}".format(data_time_name, len(date_times)))
    return data_time_name

def get_files(base_dir, index):
    """Retrieve a dictionary of filenames, including calibration(P2, P3, R, T), left_image, right_image, point_cloud, labels(could be none)

        if is_sequence:
            Will read from KITTI raw data. 
            base_dir: str, should be absolute file path to kitti_raw
            index: int, sequence number at that datetime <1> -> <2011_09_26_drive_0001_sync>
        else:
            Will read from KITTI detection data.
            base_dir: str, should be abolutie file path to <training/testing>
            index: int, index number for calib/image/velodyne
    """
    output_dict = {
        "calib":{
            "P0":None,
            "P1":None,
            "T_cam2velo":None,
            "cam_to_pose":None,
        },
        "left_image":"",
        "right_image":"",
        'key_frames': [],
        "poses":None,
        
    }
    data_pose_dir   = os.path.join(base_dir, 'data_poses')
    data_2d_raw_dir = os.path.join(base_dir, 'data_2d_raw')
    calib_dir       = os.path.join(base_dir, 'calibration')    

    sequence_name = determine_date_index(data_pose_dir, index)

    cam_calib_file = os.path.join(calib_dir, "perspective.txt")
    P0, P1 = read_P01_from_sequence(cam_calib_file)
    velo_calib_file = os.path.join(calib_dir, "calib_cam_to_velo.txt")
    T_cam2velo = read_T_from_sequence(velo_calib_file)
    cam_extrinsic_file = os.path.join(calib_dir, "calib_cam_to_pose.txt")
    T_cam2pose = read_extrinsic_from_sequence(cam_extrinsic_file)
    output_dict["calib"]["P0"] = P0
    output_dict["calib"]["P1"] = P1
    output_dict["calib"]["T_cam2velo"] = T_cam2velo
    output_dict["calib"]["cam_to_pose"] = T_cam2pose

    left_dir = os.path.join(data_2d_raw_dir, sequence_name, "image_00", "data_rgb")
    left_images = os.listdir(left_dir)
    left_images.sort()
    left_images = [os.path.join(left_dir, left_image) for left_image in left_images]

    right_dir = os.path.join(data_2d_raw_dir, sequence_name, "image_01", "data_rgb")
    right_images= os.listdir(right_dir)
    right_images.sort()
    right_images = [os.path.join(right_dir, right_image) for right_image in right_images]

    poses_file = os.path.join(data_pose_dir, sequence_name, 'poses.txt') # pose mat can be generated with official matlab toolkits
    key_frames, poses = read_poses_file(poses_file)

    output_dict["left_image"] = left_images
    output_dict["right_image"] = right_images
    output_dict["poses"] = poses
    output_dict["key_frames"] = key_frames

    return output_dict

        
