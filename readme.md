# Kitti360 Visualization

Ros package to visualize KITTI-360 data with RVIZ

## Getting Started:

Overwrite the folder names in the launch file to your data.

### Core Features:

- [x] KITTI raw data sequence support. 
- [x] Stereo RGB cameras.
- [x] LiDAR, RGB point clouds.
- [x] TF-tree (camera and LiDAR).
- [x] GUI control & ROS topic control.


## GUI

![image](docs/gui.png)

### User manual:

    index: integer selection notice do not overflow the index number.

    Stop: stop any data loading or processing of the visualization node.
    
    Pause: prevent pointer of the sequantial data stream from increasing, keep the current scene.

    Cancel: quit.

## Raw Data & Depth Prediction Dataset

We support video-like streaming raw data. Depth Prediction dataset follows similar structure of raw data, thus can be visualized in RGB point clouds together(optionally). 

![image](docs/sequence.png)

