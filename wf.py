#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Int16

#import Global


#TO DO: Tune parameters
#PID CONTROL PARAMS
kp = 1
kd = 0.02
ki = 0.0001
servo_offset = 0.0
prev_error = 0.0 
error = 0.0
integral = 0.0
delta_t = 0.1

#WALL FOLLOW PARAMS 
ANGLE_RANGE = 270 # Hokuyo 10LX has 270 degrees scan
#DESIRED_DISTANCE_RIGHT = 0.50 # meters
DESIRED_DISTANCE_LEFT = 1
VELOCITY = 0.7 # meters per second
CAR_LENGTH = 0.50 # TfollowLeftraxxas Rally is 20 inches or 0.5 meters

class WallFollow:
    """ Implement Wall Following on the car
    """
    def __init__(self):
        #Topics & Subs, Pubs
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)#: Subscribe to LIDAR
        self.flag_sub = rospy.Subscriber("/es_cmd", Int16, self.AEB_callback)
        self.drive_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, queue_size=10)#: Publish to drive

    def getRange(self, data):
        # data: single message from topic /scan
        # angle: between -45 to 225 degrees, where 0 degrees is directly to the right
        # Outputs length in meters to object with angle in lidar scan field of view # 5m 
        #make sure to take care of nans etc.
        #TODO: implement
        laser_ranges = data.ranges
        laser_ranges = laser_ranges[600:710]
        return 0.0


    def pid_control(self, error, velocity):
    
        #TODO: Use kp, ki & kd to implement a PID controller
        #       Example:
        #               drive_msg = AckermannDriveStamped()
        #               drive_msg.header.stamp = rospy.Time.now()
        #               drive_msg.header.frame_id = "laser"
        #               drive_msg.drive.speed = 0
        #               self.drive_pub.publish(drive_msg)
        
        global kp, ki, kd, delta_t, integral
        
        e_int_y = integral + 0.5*(error+prev_error)*delta_t #integrated error
        
        e_diff_y = (error - prev_error)/ delta_t #differentiated error
        
        theta_ref_un = kp* error + ki* e_int_y + kd* e_diff_y
        
        theta_max = np.radians(40)
        theta_ref = theta_ref_un
        
        integral = e_int_y
        if np.abs(theta_ref) >= theta_max:
            integral = e_int_y+ (delta_t/ki)*(theta_ref - theta_ref_un)
            theta_ref = kp* error + ki* integral + kd* e_diff_y

        speed = velocity
        steering_angle = -theta_ref

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.speed = speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)     

    def lidar_callback(self, data):

        #TO DO:  
        #       1. Get LiDAR message
        #       2. Calculate length to object with angle in lidar scan field of view
        #          and make sure to take care of nans etc.
        #       3. Based on length to object, callback 'pid_control' 

        global error, prev_error
        laser_ranges = data.ranges
        wall = 180 + (1080//270)*180
        b = laser_ranges[wall]
        
        angle = 50
        theta = (1080// 270)*angle
        a = laser_ranges[wall-theta]
        
        angle_rad = angle*np.pi/180
        # print(angle_rad)
        alpha = np.arctan2(a*np.cos(angle_rad)-b, a*np.sin(angle_rad))

        D_t = b* np.cos(alpha)

        look_up = CAR_LENGTH*np.sin(alpha)

        cd_len = D_t + look_up
        
        prev_error = error
        error = DESIRED_DISTANCE_LEFT - cd_len
        # print("error: ",DESIRED_DISTANCE_LEFT - cd_len)

        alpha_deg = np.degrees(alpha)
        # print("alpha in degree:", alpha_deg)
        if 0 <=np.abs(alpha_deg) < 10:
            # VELOCITY = 1.1
            VELOCITY = 1.1
        elif 10<= np.abs(alpha_deg) <20:
            # VELOCITY = 0.9
            VELOCITY = 1.0
        elif 20<= np.abs(alpha_deg) <30:
            # VELOCITY = 0.9
            VELOCITY = 0.7
        elif 30<= np.abs(alpha_deg) <40:
            # VELOCITY = 0.9
            VELOCITY = 0.7
        else:
            # VELOCITY = 0.7
            VELOCITY = 0.7

        self.pid_control(error, VELOCITY)

    def AEB_callback(self, data):
        if data.data == 1:
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.header.frame_id = "laser"
            drive_msg.drive.speed = 0
            drive_msg.drive.steering_angle = 0
            self.drive_pub.publish(drive_msg)  


def main(args):

    rospy.init_node("WallFollow_node", anonymous=False)
    wf = WallFollow()
    rospy.sleep(0.05)
    rospy.spin()


if __name__=='__main__':
	main(sys.argv)
