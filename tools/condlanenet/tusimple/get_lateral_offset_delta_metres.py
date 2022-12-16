#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 16:06:14 2021

@author: harish
"""

import numpy as np
import cv2
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import math
#import rospy
#from aerovect_msgs.msg import lane_offset

class LanelateralOffset():
    def __init__(self):
        #self.pub = rospy.Publisher('ego_vehicle_lane_offset', lane_offset, queue_size=10)
        self.publish_data = None

    def getEgoLane(self, dict_lanes):
        mid_x = 1280/2
        left_lane_idx = []
        right_lane_idx = []
        smallest_left_dist = 50000
        smallest_right_dist = 50000
        idx = 0
        for lane_x in dict_lanes["lanes"]:
            count = 0
            for i in reversed(lane_x):
                if (i == -2):
                    count += 1
                    continue
                else:
                    curr_lane_x_dist = i - mid_x
                    if (curr_lane_x_dist < 0 ):
                        if not left_lane_idx:
                            left_lane_idx.append(idx)
                        # left_lane_idx.append(dict_lanes["h_samples"][count])
                        elif (abs(curr_lane_x_dist) < smallest_left_dist):
                            left_lane_idx[0] = idx
                        # left_lane[1] = dict_lanes["h_samples"][count]
                            
                    if (curr_lane_x_dist > 0 ):
                        if not right_lane_idx:
                            right_lane_idx.append(idx)
                        #  right_lane.append(dict_lanes["h_samples"][count])
                        elif (abs(curr_lane_x_dist) < smallest_right_dist):
                            right_lane_idx[0] = idx
                        # right_lane[1] = dict_lanes["h_samples"][count]
                    idx += 1
                    break               
        left_lane = {"x": [], "y": []}
        right_lane = {"x": [], "y": []}
        count = 0
        if not left_lane_idx or not right_lane_idx:
            return left_lane, right_lane, left_lane_idx, right_lane_idx
        for lane_x in dict_lanes["lanes"][left_lane_idx[0]]:
            if lane_x == -2:
                count += 1
                continue
            left_lane["x"].append(lane_x)
            left_lane["y"].append(dict_lanes["h_samples"][count])
            count += 1
            
        count = 0
        for lane_x in dict_lanes["lanes"][right_lane_idx[0]]:
            if lane_x == -2:
                count += 1
                continue
            right_lane["x"].append(lane_x)
            right_lane["y"].append(dict_lanes["h_samples"][count])
            count += 1
        return left_lane, right_lane, left_lane_idx, right_lane_idx

    def getMidOfLane(self, left_lane, right_lane, left_lane_idx, right_lane_idx, dict_lanes):
        curr_coords = [0,0]
        mid_curve = []
        mid_curve_x = []
        mid_curve_y = []
        smallest_det_lane_y = left_lane["y"]
        if len(left_lane["x"]) <= len(right_lane["x"]):
            smallest_det_lane_y = left_lane["y"]
            for i in range(0,len(left_lane["x"])):
                # curr_coords[0] = math.floor((right_lane["x"][i] + left_lane["x"][i])/2)
                # curr_coords[1] = dict_lanes["h_samples"][i]
                # mid_curve_x.append(math.floor(right_lane["x"][i] + left_lane["x"][i])/2)
                # mid_curve_y.append(left_lane["y"][i])
                mid_curve.append([math.floor((right_lane["x"][i] + left_lane["x"][i])/2), left_lane["y"][i]])
        elif len(right_lane["x"]) < len(left_lane["x"]):
            smallest_det_lane_y = right_lane["y"]
            for i in range(0,len(right_lane["x"])):
                # curr_coords[0] = math.floor((right_lane["x"][i] + left_lane["x"][i])/2)
                # curr_coords[1] = dict_lanes["h_samples"][i]
                # mid_curve_x.append(math.floor(right_lane["x"][i] + left_lane["x"][i])/2)
                # mid_curve_y.append(right_lane["y"][i])
                mid_curve.append([math.floor((right_lane["x"][i] + left_lane["x"][i])/2), right_lane["y"][i]])
        return mid_curve, smallest_det_lane_y, mid_curve_x, mid_curve_y

    def getPerspectTransform(self, img,left_lane,right_lane,smallest_det_lane_y):
        src = np.float32([[left_lane["x"][0]-30,smallest_det_lane_y[0]],
            [right_lane["x"][0]+30,smallest_det_lane_y[0]],
            [right_lane["x"][-1],right_lane["y"][-1]],
            [left_lane["x"][-1],left_lane["y"][-1]],
            ])
        dst = np.float32([[150,10],[1200,10],[1200,700],[150,700]])

        M= cv2.getPerspectiveTransform(src,dst)
        return M

    def getInversePerspect(self, left_lane, right_lane, transformed_offset,smallest_det_lane_y):
        src = np.float32([[left_lane["x"][0]-30,smallest_det_lane_y[0]],
            [right_lane["x"][0]+30,smallest_det_lane_y[0]],
            [right_lane["x"][-1],right_lane["y"][-1]],
            [left_lane["x"][-1],left_lane["y"][-1]],
            ])
        dst = np.float32([[150,10],[1200,10],[1200,700],[150,700]])
        Minv = cv2.getPerspectiveTransform(dst,src)
        matrix = Minv
        normal_offset = []
        for i in range(len(transformed_offset)):
            p = transformed_offset[i]
            p = np.array([[p]], dtype=np.float32)
            p = cv2.perspectiveTransform(p, matrix)
            p_normal = (int(p[0][0][0]), int(p[0][0][1])) # after transformation
            normal_offset.append(p_normal)
        return normal_offset

    def find_curvature(self, x):
        slope = []
        dx_list = []
        dy_list = []
        for i in range(0,x.shape[1]-1):
            dx = x[0,i]-x[0,i+1]
            dy = x[1,i]-x[1,i+1]
            dx_list.append(dx)
            dy_list.append(dy)
            slope.append(dy/dx)
        curvature = np.gradient(dx_list)
        return curvature

    def getTransformedPoints(self, pts, matrix, offset_px=0):
        transformed_offset_x = []
        transformed_offset_y = []
        transformed_offset = []
        for i in range(len(pts)):
            p = pts[i]
            
            if (abs(matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]) <= 0.0001 or abs(matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]) == 0.0001):
                transformed_offset.append((0,0))
                transformed_offset_x.append(0)
                transformed_offset_y.append(0)
                good_perp_transform = False
            else:
                px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
                x_offset = px-offset_px
                p_after = (int(x_offset), int(py)) # after transformation
                transformed_offset.append(p_after)
                transformed_offset_x.append(x_offset)
                transformed_offset_y.append(py)
                good_perp_transform = True
        return good_perp_transform, transformed_offset, transformed_offset_x, transformed_offset_y

    def get_lateral_offset(self, dict_lanes, img_name):
        data = []
        kalman = cv2.KalmanFilter(4, 3)
        delta_t = 0.1
        kalman.transitionMatrix = np.array([[1., delta_t, 0., 0.], 
                                            [0., 1., 0., 0.], 
                                            [0., 0., 1., delta_t], 
                                            [0., 0., 0., 1.]], np.float32)
        kalman.measurementMatrix = np.array([[1., 0., 0., 0.],
                                            [0., 1., 0., 0.], 
                                            [0., 0., 1., 0.]], np.float32)
        
        #with open('/media/harish/T71/data_collection/atl_del_processed/Lane_Annotations/labels/test.json') as f:
        #    for line in f:
        #        data.append(json.loads(line))
        progress = 0
        offset_mtr = 0.0
        reliable_measurement = True
        bad_frame_count = 0
        spurious_frame = False
        #for dict_lanes in data:
        # if dict_lanes["raw_file"] == "1917.jpg":
        #img_name = dict_lanes["raw_file"]
        #img_name = '/media/harish/T7/data_collection/atl_del_processed/left_all_720/'+img_name
        img = cv2.imread(img_name,cv2.COLOR_BGR2RGB)     
        # img = img_name
        print("dict_lanes: \n", dict_lanes)
        egoLaneLeft, egoLaneRight, egoLaneLeftIdx, egoLaneRightIdx = self.getEgoLane(dict_lanes)
        if not egoLaneLeftIdx or not egoLaneRightIdx:
            return False
        midCurve, smallest_det_lane_y, mid_curve_x, mid_curve_y = self.getMidOfLane(egoLaneLeft, egoLaneRight, egoLaneLeftIdx, egoLaneRightIdx, dict_lanes)
        matrix = self.getPerspectTransform(img, egoLaneLeft, egoLaneRight, smallest_det_lane_y)
        result = cv2.warpPerspective(img, matrix, (1280, 720))
        # out_name = '/external_drive/data_collection/atl_del_processed/left_mid_lane_live/'+dict_lanes["raw_file"]
        
        offset_px = midCurve[-1][0]-640
        cv2.polylines(img, [np.array(midCurve,np.int32)], False, (0,0,255),4)
        ego_left_lane_list = zip(egoLaneLeft['x'], egoLaneLeft['y'])
        ego_left_lane_list = list(ego_left_lane_list)
        ego_right_lane_list = zip(egoLaneRight['x'], egoLaneRight['y'])
        ego_right_lane_list = list(ego_right_lane_list)
        good_perp_transform, transformedCam, transformedCamX, transformedCamY = self.getTransformedPoints(midCurve, matrix, offset_px)
        good_perp_transform, transformedMid, transformedMidX, transformedMidY = self.getTransformedPoints(midCurve, matrix)
        good_perp_transform, transformedLeft, transformedLeftX, transformedLeftY = self.getTransformedPoints(ego_left_lane_list, matrix)
        good_perp_transform, transformedRight, transformedRightX, transformedRightY = self.getTransformedPoints(ego_right_lane_list, matrix)
        reliable_measurement = False
        if (good_perp_transform):
            perspectiveCam = self.getInversePerspect(egoLaneLeft, egoLaneRight, transformedCam, smallest_det_lane_y)
            cv2.polylines(img, [np.array(perspectiveCam,np.int32)], False, (0,255,0),4)
            left_lane_list = zip(egoLaneLeft['x'], egoLaneLeft['y'])
            left_lane_tuple = list(left_lane_list)
            right_lane_list = zip(egoLaneRight['x'], egoLaneRight['y'])
            right_lane_tuple = list(right_lane_list)
            cv2.polylines(img, [np.array(left_lane_tuple,np.int32)], False, (255,0,0),4)
            cv2.polylines(img, [np.array(right_lane_tuple,np.int32)], False, (255,0,0),4)
            # out_name = '/home/aerovect/Documents/lane-offsets/'+dict_lanes["raw_file"]
            # print("transformedMidY: ", transformedMidY)
            mid_curve_c = np.polyfit(transformedMidY,transformedMidX, 2)
            # print("mid_curve_c: ", mid_curve_c)
            normal_off_c = np.polyfit(transformedCamY, transformedCamX, 2)
            min_lane_pts = min(len(transformedLeftY), len(transformedRightY))
            lane_width_pix_list = 0
            for i in range(0,min_lane_pts):
                lane_width_pix_list = lane_width_pix_list + (abs(transformedLeftX[i] - transformedRightX[i]))
            
            lane_width_pix = lane_width_pix_list/min_lane_pts
            mid_curve_coeff = mid_curve_c[0]*1280**2 + mid_curve_c[1]*1280 + mid_curve_c[2]
            normal_off_coeff = normal_off_c[0]*1280**2 + normal_off_c[1]*1280 + normal_off_c[2]
            offset_px1 = perspectiveCam[-1][0] -  midCurve[-1][0]
            offset_px_coeff = normal_off_coeff - mid_curve_coeff
            offset_mtr_prev = offset_mtr
            offset_mtr = offset_px1*0.005
            if abs(offset_mtr_prev - offset_mtr) > 0.7:
                offset_mtr = offset_mtr_prev
                bad_frame_count += 1
            else:
                bad_frame_count = 0
            if bad_frame_count >= 3:
                reliable_measurement = False
            else:
                reliable_measurement = True
                
            left_lane_xy = [egoLaneLeft['x'], egoLaneLeft['y']]
            right_lane_xy = [egoLaneRight['x'], egoLaneRight['y']]
            left_lane_xy = np.array(left_lane_xy)   
            right_lane_xy = np.array(right_lane_xy)
            print("\nleft_lane_xy.shape: ", left_lane_xy.shape)
            if left_lane_xy.shape[1] > 3 and right_lane_xy.shape[1] > 3:
                curvature_left = self.find_curvature(left_lane_xy)
                curvature_right = self.find_curvature(right_lane_xy)
                left_curvature_check = np.where(abs(curvature_left) >= 110)
                print("\n\nleft_curvature_check: \n", left_curvature_check)
                print("\ncurvature_left: ", curvature_left)
                right_curvature_check = np.where(abs(curvature_right) >= 110)
                if (left_curvature_check[0].shape[0] >= 2 or right_curvature_check[0].shape[0] >= 2):
                    spurious_frame = True
                else:
                    spurious_frame = False
            y_dot = 0.0
            yaw = 0.0
            prediction = kalman.predict()
            measurement = np.array([np.float32(offset_mtr), np.float32(y_dot), np.float32(yaw)])
            kalman.correct(measurement)
            #reliable_str = str(reliable_measurement)
            filtered_offset_mtr =kalman.statePost[0]
            cv2.putText(img,'Lateral offset filtered in m: '+str(filtered_offset_mtr),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.putText(img,'offset_px_coeff: '+str(offset_px_coeff),(10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img,'Lateral offset non-filtered in m: '+str(offset_mtr),(10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #cv2.putText(img,'Reliable measurement?: '+reliable_str,(10,130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img,'spurious lanes??: '+str(spurious_frame),(10,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #self.publish_data.header.stamp = rospy.Time.now()
            #self.publish_data.y_offset = filtered_offset_mtr
            #self.publish_data.reliable_measurement = reliable_measurement
            #self.pub.publish(publish_data)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            # cv2.imwrite(out_name, img)
            print("measurement:\n", measurement)
            print("offset_mtr: ", offset_mtr)
            print("filtered_offset_mtr[0].item(): ", filtered_offset_mtr[0].item())
        return filtered_offset_mtr[0].item()#, reliable_measurement
    

