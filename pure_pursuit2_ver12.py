"""
Path tracking simulation with pure pursuit steering and PID speed control.
"""
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from pandas.core.accessor import delegate_names
from scipy import io

import pandas as pd
import cubic_spline_planner
import gap_follow_lab as gf

#ROS Imports
import rospy
from sensor_msgs.msg import Image, LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray


# Parameters
k = 0.1  # look forward gain
#Lfc = 0.6  # [m] look-ahead distance
#Lfc = 0.6  # [m] look-ahead distance
Kp = 5.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 0.325  # [m] wheel base of vehicle

data = pd.read_csv('waypoint_2.csv', sep=", ")
data_xy = data[["x","y"]]
ax = data_xy["x"]
ay = data_xy["y"]

Lf = 5.0

cx, cy, _, ck, _ = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.05)

target_speed = 2.0 # [m/s]

show_animation = False

ALGORITHM = 0
#0 : pure pure_pursuit
#1 : pure_pursuit + gap_following

x_goal_finalline = [ax[len(ax)-1],ax[len(ax)-1],ax[len(ax)-1]]
y_goal_finalline = [ay[len(ay)-1],ay[len(ay)-1]+1,ay[len(ay)-1]-1]
final_goal_index = 0


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
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a

class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None
        self.show = True

        self.section_marker_array = MarkerArray()
        self.section_vis_pub = rospy.Publisher('/section_visualization', MarkerArray, queue_size =10)

    def search_target_index(self, state):
        global Lf, target_speed, ALGORITHM
        # To speed up nearest point search, doing it at only first time.
        # if self.old_nearest_point_index is None:
            # search nearest point index
        dx = [state.rear_x - icx for icx in self.cx]
        dy = [state.rear_y - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        ind = np.argmin(d)
        #print("current index:", cx[ind], cy[ind])


        # if (state.x - 1.6)**2+ ((state.y- 2.0)**2)/2 <= 1.2**2:
        #     Lf = 3
        #     target_speed = 2.5
        
        # if (state.x - 1.6)**2 +((state.y-6.2)**2)/2 <= 1.2**2:
        #     Lf = 3.5 #3.2
        #     target_speed = 0.3
    
        if (state.x - ax[1])**2+ 25*((state.y - ay[1])**2) <= 1.6**2:  #3
            Lf = 5 #1.5
            target_speed = 6.0
            
    
        if (state.x - ax[2])**2+ 25*((state.y -ay[2])**2) <= 1.4**2: #4
            Lf = 3 #3.2
            target_speed = -5.0

        if (state.x - ax[3])**2+ 25*((state.y -ay[3])**2) <= 1.4**2: #5
            Lf = 1.7 #3.2
            target_speed = 0.0
            
        if (state.x - ax[4])**2 + 25*((state.y -ay[4])**2) <= 1.4**2:  #6 
            Lf = 1.0 #1.3
            #target_speed = 0.5
            target_speed = 0.5
        
        

        if 16*((state.x - ax[5])**2)+((state.y -ay[5])**2) <= 0.8**2: #7  
            Lf = 1.1
            target_speed =2.0 #0.6
            
        
        # if (state.x -ax[7])**2 + ((state.y-ay[7])**2)/2 <= 1.2**2: #9
        #      Lf = 2.0
        #      target_speed = 2.5
        #      #print("section: 1" )
     
        if 16*(state.x +1.116)**2 + ((state.y-13.28)**2) <= 0.8**2: # between 8 and 9
                Lf = 2.2
                target_speed = 2.0
                #print("section: 1" )

        #To do

        if (state.x- ax[11])**2 + 9*(state.y-ay[11])**2 <= 1.4**2: #13
            Lf = np.hypot(abs(state.x - ax[15-2]), abs(state.y - ay[15-2]))
            target_speed = 2.5    

        # if (state.x -ax[12])**2/2 + ((state.y-ay[12])**2) <= 1.0**2: #14
        #     Lf = 1.5
        #     target_speed = 2.0
            
        # if (state.x +2.490)**2 + ((state.y-4.493)**2)/2 <= 1.2**2: #collision point 2, 15 +1
        #     Lf = 2.0
        #     target_speed = 2.0
        #     #print("section: 2" )

        if 5*(state.x-ax[13])**2 + (state.y-ay[13])**2 <= 1.2**2: #15 
            Lf = np.hypot(abs(state.x - ax[17-2]), abs(state.y - ay[17-2]))
            target_speed = 1.5
            
        
        # if 5*(state.x -ax[16-2])**2 + (state.y-ay[16-2])**2 <= 1.2**2: #after collision point, 16
        #     print("section 16: ", ax[16-2],ay[16-2])
        #     Lf = np.hypot(abs(state.x - ax[18-2]), abs(state.y - ay[18-2]))
        #     target_speed = 2.0
            
        

        if 5*(state.x -ax[17-2])**2 + ((state.y-ay[17-2])**2) <= 1.2**2: #after collision point, 17
            print("section 17: ", ax[17-2],ay[17-2])
            Lf = np.hypot(abs(state.x - ax[19-2]), abs(state.y - ay[19-2]))
            target_speed = 2.0
            
        

        if (state.x -ax[18])**2 + ((state.y-ay[18])**2)/5 <= 0.6**2: #after collision point, 19 +1 
            Lf = 2.0
            target_speed = 2.0
            #print("section: 3" )
            
        

        if (state.x -ax[19])**2 + ((state.y-ay[19])**2)/5 <= 1.2**2: # 20 +1 
            Lf = 1.7
            target_speed = 2.2
            
        
        if (state.x -ax[22])**2 + ((state.y-ay[22])**2)/5 <= 0.6**2: # 24
            Lf = 2.5
            target_speed = 2.5
            
        

        if ((state.x -ax[26])**2)/5 + ((state.y-ay[26])**2) <= 1.2**2: # 27 +1 
            Lf = 2.32
            target_speed = 2.5
            
        
        if ((state.x -ax[28])**2)/5 + ((state.y-ay[28])**2) <= 1.2**2: # 29 +1 
            Lf = 2.5
            target_speed = 2.5
            
        
        if (state.x -ax[34])**2 + ((state.y-ay[34])**2)/2 <= 1.2**2: #last point -11: # before curve
            Lf = 1.5
            target_speed = 2.0   
            
        
        #before problem point
        if (state.x -ax[36])**2 + ((state.y -ay[36])**2)/2 <= 1.2**2: #38 -> 41
            Lf = np.hypot(abs((state.x -ax[39])), abs(((state.y)-ay[39])))
            target_speed = 2.0
            
        

        if (state.x -ax[37])**2 + ((state.y-ay[37])**2)/2 <= 1.2**2: #39 -> 41
            Lf = np.hypot(abs((state.x -ax[39])), abs(((state.y)-ay[39])))
            target_speed = 0.8
            
        

        #problem point
        if (state.x-ax[38])**2 + ((state.y-ay[38])**2)/2 <= 1.2**2: #40 see #41.5
            Lf = np.hypot(abs((state.x + 8.7729)), abs(((state.y)-5.0988)))
            target_speed = 0.3 
            
        
        if (state.x -ax[39])**2 + ((state.y-ay[39])**2)/5 <= 0.8**2: #41 see # 41.5
            Lf = np.hypot(abs((state.x + 8.7729)), abs(((state.y)-5.0988)))
            target_speed = 0.3
        
        

        #if (state.x+9.101)**2 + ((state.y-4.539)**2)/10<= 0.5**2: #42 see #44
        #    Lf = np.hypot(abs((state.x + 8.229)), abs(((state.y)-3.446)))
        #    target_speed = 0.3

        if (state.x -ax[40])**2 + ((state.y-ay[40])**2)/5<= 0.5**2: #42 see #43
            Lf = np.hypot(abs((state.x -ax[41])), abs(((state.y)-ay[41])))
            target_speed = 0.3
            
        

        if (state.x -ax[41])**2 + ((state.y-ay[41])**2)/10<= 0.2**2: #43 see #45
            Lf = np.hypot(abs((state.x -ax[43])), abs(((state.y)-ax[43])))
            target_speed = 0.3
            
       

        if (state.x -ax[43])**2 + ((state.y-ay[43])**2)/10<= 0.5**2: #45 see #46
            Lf = np.hypot(abs((state.x -ax[44])), abs(((state.y)-ay[44])))
            target_speed = 0.3
            
       

        if (state.x- ax[44])**2 + ((state.y-ay[44])**2)/5 <= 0.5**2: #last point
            ALGORITHM =1
            print("0 to 1")
            
       
                    
        if 25*(state.x-ax[0])**2 + (state.y-ay[0])**2 <= 1.5**2: #the first point
            ALGORITHM =0
            print("1 or 2 to 0")  
            Lf = 3.0
            target_speed = 2.0 #+0.1*np.random.random(0,1)
            
        

        if  ALGORITHM == 1:
            Lf = np.hypot(abs((state.x -ax[0])), abs((state.y-ay[0])))
            target_speed = 0.8    

        #print("full look ahead:", Lf)

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1


        if self.show:
            self.draw_ellips(ax[1], ay[1], 1.6 ,1.6/5, 201+2) #3
            self.draw_ellips(ax[2], ay[2], 1.4 ,1.4/5, 102+2) #4
            self.draw_ellips(ax[3], ay[3], 1.4 ,1.4/5, 103+2) #5
            self.draw_ellips(ax[4], ay[4], 1.4 ,1.4/5, 104+2) #6
            self.draw_ellips(ax[5], ay[5], 0.8/4 , 0.8, 205+2) #7
            
            self.draw_ellips(-1.116, 13.29, 0.8/4,0.8, 107+2) # between 8 and 9

            self.draw_ellips(ax[11], ay[11], 1.4, 1.4/3, 111+2) # 13

            #self.draw_ellips(ax[12], ay[12], 2**0.5 , 1, 214+2) #14

            self.draw_ellips(ax[13], ay[13], 1.2/(5**0.5) ,1.2 ,113+2)
                
            #self.draw_ellips(-2.490, 4.493, 1.2 ,1.2*(2**0.5), 1160+2) between 16 and 17

            #self.draw_ellips(ax[14], ay[14], 1.2/(5**0.5) ,1.2 ,114+2) #16
            self.draw_ellips(ax[15], ay[15], 1.2/(5**0.5) ,1.2, 115+2)
            self.draw_ellips(ax[18], ay[18], 0.6 ,0.6/(5**0.5), 118+2)
            self.draw_ellips(ax[19], ay[19], 1.2 ,1.2*(5**0.5), 119+2)

            self.draw_ellips(ax[22], ay[22], 0.6 ,0.6*(5**0.5), 122+2) #24
            self.draw_ellips(ax[26], ay[26], 1.2*(5**0.5) ,1.2, 126+2)

            self.draw_ellips(ax[28], ay[28], 1.2*(5**0.5) ,1.2*(2**0.5), 128+2)

            self.draw_ellips(ax[34], ay[34], 1.2 ,1.2*(2**0.5), 134+2)
            self.draw_ellips(ax[36], ay[36], 1.2 ,1.2*(2**0.5), 136+2)
            self.draw_ellips(ax[37], ay[37], 1.2 ,1.2*(2**0.5), 137+2)
            self.draw_ellips(ax[38], ay[38], 1.2 ,1.2*(2**0.5), 138+2)

            self.draw_ellips(ax[39], ay[39], 0.8 ,0.8*(5**0.5), 139+2)
            self.draw_ellips(ax[40], ay[40], 0.5 ,0.5*(5**0.5), 140+2)
            self.draw_ellips(ax[41], ay[41], 0.2 ,0.2*(10**0.5), 141+2)
            self.draw_ellips(ax[43], ay[43], 0.5 ,0.5*(10**0.5), 143+2)
            self.draw_ellips(ax[44], ay[44], 0.5 ,0.5*(5**0.5), 144+2)
            self.draw_ellips(ax[0], ay[0], 1.5/5 ,1.5, 145+2)

            self.section_vis_pub.publish(self.section_marker_array)
            self.show = False

        return ind, Lf

    def draw_ellips(self, x_0_, y_0_, a_,b_, marker_id):
        section_marker = Marker()
        section_marker.type = Marker.LINE_STRIP
        section_marker.action = Marker.ADD
        section_marker_scale_msgs = Vector3()
        section_marker_scale_msgs.x =0.1
        section_marker_scale_msgs.y =0.1
        section_marker_scale_msgs.z =0.00
        section_marker.scale = section_marker_scale_msgs
        section_marker.pose.orientation.w = 1
        section_marker.color.g = np.random.rand()
        section_marker.color.r = np.random.rand()
        section_marker.color.b = np.random.rand()
        section_marker.color.a = 1.0
        x_0 = x_0_
        y_0 = y_0_
        a=a_
        b=b_
        delta_th = 0.1
        if a > b:
            k = (1-(b*b)/(a*a))**0.5
        else:
            k = (1-(a*a)/(b*b))**0.5
        for th in np.arange(0.0, 2*math.pi+delta_th, delta_th):
            if a> b:
                x = x_0 + (b / ( 1- (k*np.cos(th))**2 )**0.5  )* np.cos(th)
                y = y_0 + (b / ( 1- (k*np.cos(th))**2 )**0.5 ) * np.sin(th)
            else:
                x = x_0 + (a / ( 1- (k*np.sin(th))**2 )**0.5 ) * np.cos(th)
                y = y_0 + (a / ( 1- (k*np.sin(th))**2 )**0.5 ) * np.sin(th)
            point = Point()
            point.x = x
            point.y = y
            section_marker.points.append(point)

        section_marker.id = marker_id
        section_marker.header.stamp = rospy.get_rostime()
        section_marker.header.frame_id = "map"

        self.section_marker_array.markers.append(section_marker)



def pure_pursuit_steer_control(state, trajectory, pind):
    
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    #print(len(trajectory.cx))
    if ALGORITHM == 1:
        tx = x_goal_finalline[final_goal_index]
        ty = y_goal_finalline[final_goal_index]

        #print("I'm in algorithm 1!")
        #print("tx,ty:", final_goal_index)

    else:
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
state = State(x=0.0, y=0.0, yaw=0.0, v=0.0)

class pure_pursuit():
    def __init__(self):
        self.target_ind = 0
        self.lat_e = 0

        self.is_goal_clean = 0
        self.is_veiw_clean = 0

 
        #Topics & Subscriptions,Publishers
        lidarscan_topic = '/scan'
        drive_topic = '/vesc/low_level/ackermann_cmd_mux/output'
        pose_topic = '/pf/pose/odom/'
        self.drive_msg = AckermannDriveStamped()
        self.marker = Marker()
        

        self.lidar_sub = rospy.Subscriber(lidarscan_topic,LaserScan,self.lidar_callback) #None #TODO
        self.drive_pub = rospy.Publisher(drive_topic,AckermannDriveStamped,queue_size=5) #TODO
        self.target_vis_pub = rospy.Publisher('/target_visualization', Marker, queue_size=10)
        
        self.pose_sub = rospy.Subscriber(pose_topic, Odometry, self.pose_callback) 
        
        #self.spline_vis_pub = rospy.Publisher('/spline_vis_pub', Path, queue_size=10)#: Publish to drive

    def lidar_callback(self, data):
        global ALGORITHM, Lf, target_speed


        if ALGORITHM == 1: #unknown area

            self.marker.header.frame_id = 'map'
            self.marker.id = 200
            self.marker.type = self.marker.SPHERE
            self.marker.action = self.marker.ADD
            self.marker.pose.position.x = -10
            self.marker.pose.position.y = -10
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
           
            lidar_range = data.ranges
            lidar_range = np.array(lidar_range)

            self.is_goal_clean = self.is_goal_visible(lidar_range)
            self.is_veiw_clean = self.view_test(lidar_range,540)

            if not self.is_veiw_clean and not self.is_goal_clean:
                print("1_gap")

                x_goal = ax[len(ax)-1]
                y_goal = ay[len(ay)-1]

                x_dis = x_goal- state.x
                y_dis = y_goal- state.y

                angle_between = np.arctan2(y_dis, x_dis) - state.yaw
                #angle warping
                while(angle_between> np.pi):
                    angle_between -= 2*np.pi
                while(angle_between< -np.pi):
                    angle_between += 2*np.pi    

                #print("angle_between:", angle_between)
                #if abs(angle_between) <= np.radians(135):
                increment = np.radians(270) / 1080
                lidar_idx = int(1080//2+int(angle_between/increment)) #goal

                goal_l = int(lidar_idx + (np.pi/2)/increment)
                goal_r= int(lidar_idx - (np.pi/2)/increment)

                if goal_l >=5*1080 //6:
                    goal_l = 5*1080//6

                if goal_r <= 1080//6:
                    goal_r = 1080//6   

                print("goal_l, goal_r:", goal_l, goal_r)     

                
                #lidar_range[0: back+1] = 1000
                #idar_range[1080 - back: 1080] = 1000
                lidar_range = lidar_range[goal_r:goal_l]
                lidar_len = len(lidar_range)

                min_distance = np.min(lidar_range)
                closest = np.argmin(lidar_range) 

                #print("min distance: ", min_distance)

                bubble_r = 50 #100
                bubble_r_clo = 55 #110
                bubble_start_clo = closest - bubble_r_clo
                bubble_end_clo = closest - bubble_r_clo
                lidar_range[bubble_start_clo:bubble_end_clo] = 1000


                for i in range(0,lidar_len):
                    if lidar_range[i] < 1.3: #1.1
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

                print("num of free:",len(free_space))        

                start_idx, end_idx=self.find_max_gap(free_space)
                #print("start: ", start_idx, "end: ", end_idx)

                lidar_range = lidar_range * cut_bound

                best = self.find_best_point(start_idx, end_idx, lidar_range)

                best =  goal_r + best
                print("best: ", best)

                steering_angle_ref = (270 / 1080) *(best - 540)
                steering_angle = steering_angle_ref
                if min_distance < 0.5:
                    steering_angle = 2* steering_angle #/ (min_distance*2)    
                direction = np.sign(steering_angle)

                if np.abs(steering_angle) < 1:
                    steering_angle = 0

                if np.abs(steering_angle) >45 :
                    steering_angle = direction *45

                #print("steering_angle in degree:", steering_angle)
                if 0 <=np.abs(steering_angle) < 5:
                    VELOCITY = 0.8 #1.3
                elif 5<= np.abs(steering_angle) <10:
                    VELOCITY = 0.8 #1.1
                else:
                    VELOCITY = 0.8 #0.9

                speed = VELOCITY
                steering_angle = np.radians(steering_angle)

                self.drive_msg.header.stamp = rospy.Time.now()
                self.drive_msg.drive.steering_angle = steering_angle
                self.drive_msg.drive.speed = speed + 0.1*np.random.rand()
                self.drive_pub.publish(self.drive_msg)

        if 25*(state.x-ax[0])**2 + ((state.y-ay[0])**2) <= 1.5**2: #the first point 
            ALGORITHM = 0
            print("1 or 2 to 0")
            Lf = 3.0
            target_speed = 2.0

        #print("Lf:", Lf)                

    def pose_callback(self, data):
        state.x = data.pose.pose.position.x
        state.y = data.pose.pose.position.y
        state.v = data.twist.twist.linear.x
        #print("state_speed:", state.v)

        x = data.pose.pose.orientation.x
        y = data.pose.pose.orientation.y
        z = data.pose.pose.orientation.z
        w = data.pose.pose.orientation.w

        _, _, yaw = euler_from_quaternion(x, y, z, w)
        state.yaw = yaw
        #print("state_yaw:", state.yaw)

        if ALGORITHM == 0:
            print("0")
            self.target_ind,_= target_course.search_target_index(state)
            
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

            
            #print("target_index:", cx[self.target_ind], cy[self.target_ind])

            ai = proportional_control(target_speed, state.v)
            speed = state.v + ai*dt
            #print("speed:", speed)

            steer, self.target_ind = pure_pursuit_steer_control(state, target_course, self.target_ind)
            #print("steering_angle:", steer)

            state.update(ai,steer)

            self.drive_msg.header.stamp = rospy.Time.now()
            self.drive_msg.drive.steering_angle = steer
            self.drive_msg.drive.speed = speed
            self.drive_pub.publish(self.drive_msg)

        elif ALGORITHM == 1:
            
           
            if self.is_goal_clean or self.is_veiw_clean:

                self.target_ind,_= target_course.search_target_index(state)
            
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

                print("1_pursuit")
                ai = proportional_control(target_speed, state.v)
                speed = state.v + ai*dt
                print("speed, Lf:", speed, Lf)
                steer, self.target_ind = pure_pursuit_steer_control(state, target_course, self.target_ind)
                #print("steering_angle:", steer)

                state.update(ai,steer)

                self.drive_msg.header.stamp = rospy.Time.now()
                self.drive_msg.drive.steering_angle = steer
                self.drive_msg.drive.speed = speed
                self.drive_pub.publish(self.drive_msg)


        self.target_vis_pub.publish(self.marker)
        

 

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

    def is_goal_visible(self,lidar):
        global final_goal_index

        goal_clean = False

        distance = [999]*len(x_goal_finalline)
        measured = -100 #no measure, no scene

        for i in range(0,len(x_goal_finalline)): 
            x_dis = x_goal_finalline[i]- state.x
            y_dis = y_goal_finalline[i]- state.y
            expected = (x_dis**2 + y_dis**2)**0.5

            angle_between = np.arctan2(y_dis, x_dis) - state.yaw
            #angle warping
            while(angle_between> np.pi):
                angle_between -= 2*np.pi
            while(angle_between< -np.pi):
                angle_between += 2*np.pi    

            #print("angle_between:", angle_between)
            if abs(angle_between) <= np.radians(135):
                increment = np.radians(270) / 1080
                lidar_idx = int(1080//2+int(angle_between/increment)) #goal
                measured = lidar[lidar_idx]
                #print("expected:", expected)
                #print("measured: ", measured)

                goal_clean = self.view_test(lidar,lidar_idx)

                if expected < measured  and goal_clean:
                    #print("current state:", expected, lidar[lidar_idx-10],measured, lidar[lidar_idx+10], final_goal_index)
                    
                    distance[i] = expected
                    goal_clean = True

        #print("distance:", distance)
        final_goal_index = np.argmin(distance)

        return goal_clean             

    def view_test(self,lidar, lidar_idx_c):
        increment = np.radians(270 / 1080)
        safe_distance = 0.65 #0.7
        expected = 1.3 #1.3

        theta = np.arctan2(safe_distance/2,expected)

        #angle warping
        while(theta> np.pi):
            theta-= 2*np.pi
        while(theta< -np.pi):
            theta += 2*np.pi

        lidar_index_r = int(lidar_idx_c - theta//increment +1)
        lidar_index_l = int(lidar_idx_c + theta//increment +1)

        lidar_idx_bound_l_f = int((lidar_idx_c + (np.pi/2)/increment))
        lidar_idx_bound_r_f = int((lidar_idx_c - (np.pi/2)/increment))

        lidar_idx_bound_l_r = int((lidar_idx_c + (3*np.pi/4)/increment))
        lidar_idx_bound_r_r = int((lidar_idx_c - (3*np.pi/4)/increment))

        if lidar_index_l > 1080:
            lidar_index_l= 1079
    
        if lidar_index_r < 0:
            lidar_index_r = 0
    
        if lidar_idx_bound_l_f > 1080:
            lidar_idx_bound_l_f= 1079
    
        if lidar_idx_bound_r_f < 0:
            lidar_idx_bound_r_f = 0

        if lidar_idx_bound_l_r > 1080:
            lidar_idx_bound_l_r= 1079
    
        if lidar_idx_bound_r_r < 0:
            lidar_idx_bound_r_r = 0
        
        #view = True
        for i in range(lidar_index_r,lidar_index_l):
            tmp_theta = (i-lidar_idx_c)*increment
            if abs(expected/np.cos(tmp_theta)) > lidar[i]:
                # print("Don't go there! - front")            
                return False
                

        for i in range(lidar_index_l, lidar_idx_bound_l_f):
            tmp_theta = (i-lidar_idx_c)*increment
            if abs(lidar[i]*np.sin(tmp_theta)) < safe_distance/2:
                # print("Don't go there! - leftside")
                return False

        for i in range(lidar_idx_bound_r_f, lidar_index_r):
            tmp_theta = (i-lidar_idx_c)*increment
            if abs(lidar[i]*np.sin(tmp_theta)) < safe_distance /2:
                # print("Don't go there! - rightside")
                return False
                #view = False

        safe_distance_rear = 0.75
        for i in range(lidar_idx_bound_r_r, lidar_idx_bound_r_f):
            tmp_theta = (i-lidar_idx_c)*increment
            if abs(lidar[i]*np.sin(tmp_theta)) < safe_distance_rear /2:
                # print("Don't go there! - rightback")
                return False
                #view = False

        for i in range(lidar_idx_bound_l_f, lidar_idx_bound_l_r):
            tmp_theta = (i-lidar_idx_c)*increment

            #if i == (lidar_idx_bound_l_f + lidar_idx_bound_l_r) //2:
            #    print("lidar:", tmp_theta, lidar[i], lidar[i]*np.sin(tmp_theta))
            if abs(lidar[i]*np.sin(tmp_theta)) < safe_distance_rear /2:
            #    print("Don't go there! - leftback")
                return False
                #view = False
        

        return True
        #return view
    
def main(args):
    rospy.init_node("Pure_Pursuit", anonymous=True)
    PP = pure_pursuit()

    rospy.sleep(0.1)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)