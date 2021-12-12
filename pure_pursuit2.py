"""
Path tracking simulation with pure pursuit steering and PID speed control.
"""
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import io

import pandas as pd
import cubic_spline_planner

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker  


# Parameters
k = 0.1  # look forward gain
#Lfc = 0.6  # [m] look-ahead distance
#Lfc = 0.6  # [m] look-ahead distance
Kp = 5.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 0.325  # [m] wheel base of vehicle

data = pd.read_csv('waypoint_2.csv', sep=", ")
data_xy = data[["x","y"]]
cx = data_xy["x"]
cy = data_xy["y"]

# transition = pd.read_csv('transition.csv', sep=", ")
# tran_x = transition["x"]
# tran_y = transition["y"]
# tran_Lfc = transition["Lfc"]

#curr_ind = 0
#Lfc = tran_Lfc[curr_ind]
Lf = 5.0

cx, cy, _, ck, _ = cubic_spline_planner.calc_spline_course(cx, cy, ds=0.05)


target_speed = 2.0 # [m/s]

show_animation = False

import math



def euler_from_quaternion(x, y, z, w):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
    
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)
    
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
    
		return roll_x, pitch_y, yaw_z # in radians

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):
        global Lf, target_speed
        # To speed up nearest point search, doing it at only first time.
        # if self.old_nearest_point_index is None:
            # search nearest point index
        dx = [state.rear_x - icx for icx in self.cx]
        dy = [state.rear_y - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)
        print("current index:", cx[ind], cy[ind])


        # if (state.x - 1.6)**2+ ((state.y- 2.0)**2)/2 <= 1.2**2:
        #     Lf = 3
        #     target_speed = 2.5
        
        # if (state.x - 1.6)**2 +((state.y-6.2)**2)/2 <= 1.2**2:
        #     Lf = 3.5 #3.2
        #     target_speed = 0.3
    
        if (state.x - 1.868)**2+ ((state.y-2.020)**2)/2 <= 1.2**2: 
            Lf = 5 #1.5
            target_speed = 3.0

        if (state.x - 1.451)**2+ ((state.y-6.702)**2)/2 <= 1.2**2: 
            Lf = 3 #3.2
            target_speed = 2.8

        if (state.x - 1.98)**2 + ((state.y-12.04)**2)/2 <= 1.2**2:  #before corner
            Lf = 1.3 #1.0
            target_speed = 0.3

        if (state.x - 0.980)**2/1.2 +((state.y-13.28)**2) <= 0.8**2: #1.2 #atfer corner
            Lf = 1.0
            target_speed = 0.4

        if (state.x +1.808)**2 + ((state.y-13.17)**2)/2 <= 1.2**2: 
             Lf = 2.0
             target_speed = 2.5

        if (state.x +4.889)**2/2 + ((state.y-6.661)**2) <= 1.0**2: #collision point,13 +1
             Lf = 1.5
             target_speed = 2.0

        if (state.x +2.490)**2 + ((state.y-4.493)**2)/2 <= 1.2**2: #collision point 2, 15 +1
             Lf = 2.0
             target_speed = 2.0

        if (state.x +6.926)**2 + ((state.y-4.274)**2)/5 <= 1.2**2: #after collision point, 19 +1 
             Lf = 2.0
             target_speed = 2.0

        if (state.x +7.442)**2 + ((state.y-5.740)**2)/5 <= 1.2**2: # 20 +1 
             Lf = 1.7
             target_speed = 2.2

        if (state.x +6.349)**2 + ((state.y-9.566)**2)/5 <= 1.2**2: # 23 +1
            Lf = 2.5
            target_speed = 2.5

        if ((state.x +8.498)**2)/5 + ((state.y-16.48)**2) <= 1.2**2: # 27 +1 
            Lf = 2.32
            target_speed = 2.5

        if ((state.x +12.32)**2)/5 + ((state.y-16.15)**2) <= 1.2**2: # 29 +1 
            Lf = 2.5
            target_speed = 2.5

        if (state.x +12.30)**2 + ((state.y-11.86)**2)/2 <= 1.2**2: #last point -11: # before curve
             Lf = 1.5
             target_speed = 2.0   

        if (state.x+9.343)**2 + ((state.y-8.532)**2)/2 <= 1.2**2: #last point -8
            Lf = 2.0
            target_speed = 2.0      

        if (state.x+8.929)**2 + ((state.y-7.036)**2)/2 <= 1.2**2: #last point -7
            Lf = 1.6
            target_speed = 0.78 

        if (state.x+8.777)**2 + ((state.y-5.474)**2)/2 <= 0.8**2: #last point -6 
            Lf = 0.98
            target_speed = 0.3

        if (state.x+7.653)**2 + ((state.y-2.488)**2)/5 <= 0.5**2: #last point 
            Lf = 2.2
            target_speed = 2.5
        
        if (state.x-0.014)**2/2 + (state.y-0.628)**2 <= 1.2**2: #the first point 
            Lf = 3.0
            target_speed = 2.0 

        print("full look ahead:", Lf)

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    #print(len(trajectory.cx))
    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    #delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)
    delta = math.atan2(2.0 * math.sin(alpha) / Lf, 1.0)

    return delta, ind

target_course = TargetCourse(cx, cy)

class pure_pursuit():
    def __init__(self):

        self.state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)
        
        self.target_ind = 0

        self.lat_e = 0
 
        #Topics & Subscriptions,Publishers
        #lidarscan_topic = '/scan'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/output'
        pose_topic = '/pf/pose/odom/'
        self.drive_msg = AckermannDriveStamped()
        self.marker = Marker()

        #self.lidar_sub = rospy.Subscriber(lidarscan_topic,LaserScan,self.lidar_callback) #None #TODO
        self.drive_pub = rospy.Publisher(drive_topic,AckermannDriveStamped,queue_size=5) #TODO
        self.target_vis_pub = rospy.Publisher('/target_visualization', Marker, queue_size=10)
        self.pose_sub = rospy.Subscriber(pose_topic, Odometry, self.pose_callback) 
        #self.spline_vis_pub = rospy.Publisher('/spline_vis_pub', Path, queue_size=10)#: Publish to drive

    def pose_callback(self, data):
        self.state.x = data.pose.pose.position.x
        self.state.y = data.pose.pose.position.y
        self.state.v = data.twist.twist.linear.x
        print("state_speed:", self.state.v)

        x = data.pose.pose.orientation.x
        y = data.pose.pose.orientation.y
        z = data.pose.pose.orientation.z
        w = data.pose.pose.orientation.w

        _, _, yaw = euler_from_quaternion(x, y, z, w)
        self.state.yaw = yaw
        print("state_yaw:", self.state.yaw)

        self.target_ind,_= target_course.search_target_index(self.state)

        
        self.marker.header.frame_id = 'map'
        self.marker.id = 1
        self.marker.type = self.marker.SPHERE
        self.marker.action = self.marker.ADD
        self.marker.pose.position.x = cx[self.target_ind]
        self.marker.pose.position.y = cy[self.target_ind]
        self.marker.pose.position.z = 0
        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z = 0
        self.marker.pose.orientation.w = 1
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0.5
        self.marker.color.a = 1
        self.marker.color.r = 0.9
        self.marker.color.g = 0.0
        self.marker.color.b = 0.2

        
        print("target_index:", cx[self.target_ind], cy[self.target_ind])

        ai = proportional_control(target_speed, self.state.v)
        speed = self.state.v + ai*dt
        print("speed:", speed)

        steer, self.target_ind = pure_pursuit_steer_control(self.state, target_course, self.target_ind)
        #steer = self.state.v / WB * math.tan(di)
        print("steering_angle:", steer)

        #if (self.state.x - 1.4)**2 +  ((self.state.y-13.5)**2)/2 <= 1.2**2:
        #    steer = 0.1

        # if (self.state.x +2.388)**2 + ((self.state.y-12.29)**2)/2 <= 1.2**2: 
        #     steer = 0.25

        self.state.update(ai,steer)

        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_msg.drive.steering_angle = steer
        self.drive_msg.drive.speed = speed
        self.drive_pub.publish(self.drive_msg)

        self.target_vis_pub.publish(self.marker)

        # pose_msgs.pose.position.x = ix
        # pose_msgs.pose.position.y = iy

# def spline_talker():
#     #rospy.init_node("spline_vis", anonymous=False)

#     spline_vis_pub = rospy.Publisher('/spline_vis_pub', Path, queue_size=10)#: Publish to drive
#     path_msg = Path()
#     path_msg.header.frame_id = "map"
#     path_msg.header.stamp = rospy.Time.now()

#     c_len = len(cx)
#     for n in range(0,c_len):
#         pose_msgs = PoseStamped()
#         pose_msgs.pose.position.x = cx[n]
#         pose_msgs.pose.position.y = cy[n]
#         pose_msgs.pose.position.z = 0.0
#         pose_msgs.pose.orientation.x = 0.0
#         pose_msgs.pose.orientation.y = 0.0
#         pose_msgs.pose.orientation.z = 0.0
#         pose_msgs.pose.orientation.w = 1.0
#         path_msg.poses.append(pose_msgs)

#     while(True):   
#         spline_vis_pub.publish(path_msg)

def main(args):
    rospy.init_node("Pure_Pursuit", anonymous=True)
    PP = pure_pursuit()

    #spline_talker()
    
    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)