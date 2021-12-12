#!/usr/bin/env python
from __future__ import print_function
import sys
import math
import numpy as np
import itertools,operator

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive



prev_steering_angle = 0
t_prev = 0.0
prev_range = []

class reactive_follow_gap:
    def __init__(self):
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/output'
        self.drive_msg = AckermannDriveStamped()
        self.lidar_sub = rospy.Subscriber(lidarscan_topic,LaserScan,self.lidar_callback) #None #TODO
        self.drive_pub = rospy.Publisher(drive_topic,AckermannDriveStamped,queue_size=5) #TODO
    	

    def find_max_gap(self, free_space_ranges):
        """ Return the start index & end index of the max gap in free_space_ranges
        """
        max_len = -100
        start_idx = 0

        for free_space in free_space_ranges:
            if max_len < free_space[0]:
                max_len = free_space[0]
                start_idx = free_space[1]        

        return start_idx, start_idx+ max_len    
    
    def find_best_point(self, start, end, lidar_ranges):
        """Start_i & end_i are start and end indicies of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        
        best = (start+end)//2
        # ranges = lidar_ranges[start: end]
        # best = np.argmax(ranges)

        # capacity = 130

        # if best != 0 and lidar_ranges[best] - lidar_ranges[best-10] > 2:
        #     best += capacity
        # elif best != 1079 and lidar_ranges[best] - lidar_ranges[best+10] > 2:
        #     best -= capacity
        # elif best == start:
        #     best += 100
        # elif best == end:
        #     best -= 100

        return best

    def lidar_callback(self, data):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """

        #TO DO:  
        #       1. Get LiDAR message
        #       2. Find closest point to LiDAR
        #       3. Eliminate all points inside 'bubble' (set them to zero) 
        #       4. Fine the max gap
        #       4. Find the best point
        #       5. Publish Drive message
        
        lidar_range = data.ranges
        lidar_range = np.array(lidar_range)

        lidar_len = len(lidar_range)
        back = lidar_len //6
        lidar_range[0: back+1] = 1000
        lidar_range[1080 - back: 1080] = 1000
         
        min_distance = np.min(lidar_range)
        closest = np.argmin(lidar_range) 

        print("min distance: ", min_distance)
            
        bubble_r = 100
        bubble_r_clo = 110
        bubble_start_clo = closest - bubble_r_clo
        bubble_end_clo = closest - bubble_r_clo
        lidar_range[bubble_start_clo:bubble_end_clo] = 1000


        for i in range(0,lidar_len):
            if lidar_range[i] < 1.1:
                bubble_start = i - bubble_r
                bubble_end = i + bubble_r

                if bubble_start < 0:
                     bubble_start = 0
                if bubble_end > lidar_len:
                    bubble_end = lidar_len

                lidar_range[bubble_start:bubble_end] = 1000

        cut_bound = lidar_range < 1000

        gap_len = 0
        started = False
        free_space = []
        for i in range(0, lidar_len):
            if cut_bound[i] == True:
                gap_len += 1
                started = True
            elif cut_bound[i] == False and started == True:
                free_space.append([gap_len, i-gap_len])
                gap_len = 0
                started = False
            
            if i == lidar_len -1 and started ==True:
                free_space.append([gap_len, i- gap_len+1])

        start_idx, end_idx=self.find_max_gap(free_space)
        print("start: ", start_idx, "end: ", end_idx)

        lidar_range = lidar_range * cut_bound

        best = self.find_best_point(start_idx, end_idx, lidar_range)

        #best = start_idx + best
        print("best: ", best)

        steering_angle_ref = (270 / 1080) *(best - 540)
        steering_angle = steering_angle_ref
        if min_distance < 0.3:
            steering_angle = steering_angle / (min_distance*3)    
        direction = np.sign(steering_angle)

        if np.abs(steering_angle) < 1:
            steering_angle = 0

        if np.abs(steering_angle) >45 :
            steering_angle = direction *45

        print("steering_angle in degree:", steering_angle)
        if 0 <=np.abs(steering_angle) < 5:
            VELOCITY = 1.3
        elif 5<= np.abs(steering_angle) <10:
            VELOCITY = 1.1
        else:
            VELOCITY = 0.9

        speed = VELOCITY
        steering_angle = np.radians(steering_angle)
        
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.drive.steering_angle = steering_angle
        self.drive_msg.drive.speed = speed
        self.drive_pub.publish(self.drive_msg)


def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = reactive_follow_gap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
