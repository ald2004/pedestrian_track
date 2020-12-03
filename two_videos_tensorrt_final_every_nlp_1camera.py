#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 09:55:26 2019

@author: BOE
"""

from __future__ import division, print_function, absolute_import
# from timeit import time
import warnings
import cv2
import numpy as np
from numpy import *
from PIL import Image
import os
import math
from sklearn.utils.linear_assignment_ import linear_assignment
import time
import json
import redis
from Yolov3_TensorRT.yolov3_tensorrt import yolov3
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')

import yaml

W = 15.0
H = 6.0


def box_linear_interpolation(box1, box2, inter_num):
    dist = box2 - box1
    dist_uint = dist / (inter_num + 1)
    inter_boxes = []
    for i in range(inter_num):
        inter_box = box1 + dist_uint * (i + 1)
        inter_boxes.append(inter_box)
    return inter_boxes


def convert_coordinate(H_mat, bbox):
    foot_point = mat([(bbox[0] + bbox[2]) / 2, bbox[3], 1])
    point_xy = H_mat * (foot_point.transpose())
    point_xy_arr = np.array(point_xy)
    x = min(max(point_xy_arr[0][0] / point_xy_arr[2][0], 0), W)
    y = min(max(point_xy_arr[1][0] / point_xy_arr[2][0], 0), H)
    return x, y


matched_list = []  # value is the ID of each person
id1_list = []
id2_list = []


def is_matched(list, id):
    if id < list.__len__():
        if list[id] >= 0:
            return True
    return False


def combine_video(features_1, features_2, boxes1_xy, boxes2_xy, track_boxes_1, track_boxes_2):
    xy_threshold = 2.0
    cos_threshold = 0.3
    final_threshold = 3.0

    matches, unmatched_1, unmatched_2 = [], [], []

    if boxes1_xy.shape[0] <= 0:
        unmatched_2 = range(boxes2_xy.shape[0])
        return matches, unmatched_1, unmatched_2
    if boxes2_xy.shape[0] <= 0:
        unmatched_1 = range(boxes1_xy.shape[0])
        return matches, unmatched_1, unmatched_2

    xy_cost_matrix = np.zeros((boxes1_xy.shape[0], boxes2_xy.shape[0]), dtype=np.float32)
    for mm in range(boxes1_xy.shape[0]):
        for nn in range(boxes2_xy.shape[0]):
            xy_cost_matrix[mm, nn] = np.power(boxes1_xy[mm, 0] - boxes2_xy[nn, 0], 2) + np.power(
                boxes1_xy[mm, 1] - boxes2_xy[nn, 1], 2)
            xy_cost_matrix[mm, nn] = math.sqrt(xy_cost_matrix[mm, nn])
            if (xy_cost_matrix[mm, nn] > xy_threshold):
                xy_cost_matrix[mm, nn] = 1e+5
            else:
                xy_cost_matrix[mm, nn] = 0

    features_1 = np.asarray(features_1) / np.linalg.norm(features_1, axis=1, keepdims=True)
    features_2 = np.asarray(features_2) / np.linalg.norm(features_2, axis=1, keepdims=True)
    cosine_cost_matrix = 1. - np.dot(features_1, features_2.T)
    for ii in range(cosine_cost_matrix.shape[0]):
        for jj in range(cosine_cost_matrix.shape[1]):
            if cosine_cost_matrix[ii, jj] > cos_threshold:
                cosine_cost_matrix[ii, jj] = 1e+5

    # indices = linear_assignment(cosine_cost_matrix)
    final_cost_matrix = xy_cost_matrix + 1.0 * cosine_cost_matrix

    # remove matched ids
    for mm in range(boxes1_xy.shape[0]):
        id1 = int(track_boxes_1[mm][-1])
        if is_matched(id1_list, id1):
            id2 = matched_list[id1_list[id1]][1]
            for nn in range(boxes2_xy.shape[0]):
                if id2 == track_boxes_2[nn][-1]:
                    final_cost_matrix[mm, :] = 1e+5
                    final_cost_matrix[:, nn] = 1e+5
                    matches.append([mm, nn])
                    break

    # assignment
    indices = linear_assignment(final_cost_matrix)

    # for idx in range(boxes1_xy.shape[0]):
    #    if idx not in indices[:, 0]:
    #        unmatched_1.append(idx)
    # for idx in range(boxes2_xy.shape[0]):
    #    if idx not in indices[:, 1]:
    #        unmatched_2.append(idx)
    for idx in range(boxes1_xy.shape[0]):
        tmp_list = []
        if len(matches) > 0:
            for i in range(len(matches)):
                tmp_list.append(matches[i][0])
        if idx not in tmp_list:
            if idx not in indices[:, 0]:
                unmatched_1.append(idx)

    for idx in range(boxes2_xy.shape[0]):
        tmp_list = []
        if len(matches) > 0:
            for i in range(len(matches)):
                tmp_list.append(matches[i][1])
        if idx not in tmp_list:
            if idx not in indices[:, 1]:
                unmatched_2.append(idx)
    for row, col in indices:
        if final_cost_matrix[row, col] > final_threshold:
            tmp_row_list = []
            tmp_col_list = []
            if len(matches) > 0:
                for i in range(len(matches)):
                    tmp_row_list.append(matches[i][0])
                    tmp_col_list.append(matches[i][1])
            if (row not in tmp_row_list):
                unmatched_1.append(row)
            if (col not in tmp_col_list):
                unmatched_2.append(col)
        else:
            matches.append([row, col])
            id1 = int(track_boxes_1[row][-1])
            id2 = int(track_boxes_2[col][-1])
            if is_matched(id1_list, id1) & (not is_matched(id2_list, id2)):
                matched_list[id1_list[id1]][1] = id2
            elif is_matched(id2_list, id2):
                matched_list[id2_list[id2]][0] = id1
            else:
                matched_list.append([id1, id2])
                while id1_list.__len__() <= id1:
                    id1_list.append(-1)
                while id2_list.__len__() <= id2:
                    id2_list.append(-1)
                id1_list[id1] = matched_list.__len__() - 1
                id2_list[id2] = matched_list.__len__() - 1
    # remove repeat index
    unmatched_1 = list(set(unmatched_1))
    unmatched_2 = list(set(unmatched_2))

    return matches, unmatched_1, unmatched_2


def main():
    yml_name = '../config_files/pedestrian_cfg.yaml'
    f_yml = open(yml_name, 'r')
    cfg = f_yml.read()
    f_yml.close()
    config = yaml.load(cfg)
    # add timestamp to unique each same ID every restart
    timeID = time.strftime('%H%M', time.localtime()) + '_'
    # open two videos
    # vid_out_path_1 = '/home/wang/src/camera_1_1.mp4'
    # vid_out_path_2 = '/home/wang/src/camera_2_1.mp4'

    vid_out_path_1 = config['camera_path'][0]
    vid_out_path_2 = config['camera_path'][1]
    video_capture_1 = cv2.VideoCapture(vid_out_path_1)
    video_capture_2 = cv2.VideoCapture(vid_out_path_2)

    # Definition of the parameters
    max_cosine_distance = 0.25
    # feature num for each track, default none
    nn_budget = 30

    # camera's Homography matrix path
    H_file_path_1 = config['camera_calibration'][0]
    H_file_path_2 = config['camera_calibration'][1]
    H_1 = np.loadtxt(H_file_path_1)
    H_mat_1 = mat(H_1)
    H_2 = np.loadtxt(H_file_path_2)
    H_mat_2 = mat(H_2)
    ###############################################################################
    #
    show_image = config['show_image_flag']
    writeVideo_flag = config['write_video_flag']
    write_image_flag = config['write_image_flag']
    write_redis = config['write_redis_flag']

    w_1 = int(video_capture_1.get(3))
    h_1 = int(video_capture_1.get(4))
    w_2 = int(video_capture_2.get(3))
    h_2 = int(video_capture_2.get(4))
    # write two results video if need
    if writeVideo_flag:
        # Define the codec and create VideoWriter object

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_1 = cv2.VideoWriter('output_1.avi', fourcc, 25, (w_1, h_1))
        list_file_1 = open('detection_1.txt', 'w')
        frame_index = -1

        out_2 = cv2.VideoWriter('output_2.avi', fourcc, 25, (w_2, h_2))
        list_file_2 = open('detection_2.txt', 'w')

    if write_image_flag:
        frame_index = -1
        image_path_1 = config['write_image_out'][0]
        image_path_2 = config['write_image_out'][1]

    if write_redis:
        rds_host = config['redis_host']
        rds_port = config['redis_port']
        rds = redis.Redis(host=rds_host, port=rds_port)

        rds_skip_frame = config['rds_skip_frame']  # redis frame inter
    mix_skip = config['mix_skip']
    # origin re-ID model
    model_filename = 'model_data/mars-small128.pb'
    # model_filename = 'model_data/reid_resnet_trt32_fp32.engine'
    encoder = gdet.create_box_encoder(model_filename, batch_size=32, two_images=True)

    metric1 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    #    tracker_1 = Tracker(metric1, entrance=[0, 0, 200, 720])
    #    tracker_2 = Tracker(metric2, entrance=[0, 0, 200, 720])

    tracker_1 = Tracker(metric1, entrance=None)
    tracker_2 = Tracker(metric2, entrance=None)
    # define variables for skip frames
    process_time = []

    frame_buf_1 = []
    boxes_buf_1 = []
    track_buf_feature_1 = []
    detections_buf_1 = []

    frame_buf_2 = []
    boxes_buf_2 = []
    track_buf_feature_2 = []
    detections_buf_2 = []

    # interval number between two detected frame
    frame_interval = 3
    # buffer size
    frame_num_thd = frame_interval + 2

    i = 0
    j = 0
    if show_image:
        cv2.namedWindow('camera1', 0)
        cv2.namedWindow('camera2', 0)
        cv2.resizeWindow('camera1', 640, 360)
        cv2.resizeWindow('camera2', 640, 360)
        cv2.moveWindow('camera1', 100, 100)
        cv2.moveWindow('camera2', 800, 100)

    for iii in range(33):
        ret_1, frame_1 = video_capture_1.read()
    ret_2, frame_2 = video_capture_2.read()

    while ret_1 and ret_2:
        t1 = time.time()
        ret_1, frame_1 = video_capture_1.read()
        # ret_2, frame_2 = video_capture_2.read()
        if ret_1 != True or ret_2 != True:
            if i > 100:
                video_capture_1.release()
                # video_capture_2.release()
                video_capture_1 = cv2.VideoCapture(vid_out_path_1)
                # video_capture_2 = cv2.VideoCapture(vid_out_path_2)
                ret_1, frame_1 = video_capture_1.read()
                # ret_2, frame_2 = video_capture_2.read()
            else:
                print('No Video Data')
                break
        frame_time = time.localtime()
        #        image_1 = Image.fromarray(frame_1)
        frame_buf_1.append(frame_1)
        #        image_2 = Image.fromarray(frame_2)
        frame_buf_2.append(frame_2)
        ###############################################################################
        #                                  buffer
        ###############################################################################
        # every frame_interval + 1 frame detected
        if i % (frame_interval + 1) == 0:
            # yolo detect features and boxes
            # boxes_1 = yolo.detect_image(frame_1)
            boxes_1 = yolov3.detect_frame(frame_1, w_1, h_1)
            boxes_2 = []  # yolov3.detect_frame(frame_2, w_2, h_2)

            t2 = time.time()
            # boxes_2 = yolo.detect_image(image_2)
            # features_1 = encoder(frame_1, boxes_1)
            # features_2 = encoder(frame_2, boxes_2)
            features_1, features_2 = encoder(frame_1, boxes_1, frame_2, boxes_2)
            print("encoder time = ", time.time() - t2)

            # score to 1.0 here.
            '''
            got boxes and features
            '''
            detections_1 = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes_1, features_1)]
            detections_2 = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes_2, features_2)]
            # tracker predict and update
            tracker_1.predict()
            tracker_1.update(detections_1, frame_1)
            tracker_2.predict()
            tracker_2.update(detections_2, frame_2)
            # extract track1 boxes
            detections_buf_1.append(detections_1)
            boxes_final_1 = np.empty([0, 5])
            track_feature_1 = []
            for n, track in enumerate(tracker_1.tracks):
                if (not track.is_confirmed()) or track.time_since_update > 1:
                    continue
                box = np.reshape(np.int32(track.to_tlbr()), [1, 4])
                box_id = np.array([[track.confirm_id]])
                box = np.append(box, box_id, 1)
                boxes_final_1 = np.append(boxes_final_1, box, 0)
                track_feature_1.append(track.last_feature)
            boxes_buf_1.append(boxes_final_1)
            track_buf_feature_1.append(track_feature_1)
            # extract track2 boxes
            detections_buf_2.append(detections_2)
            boxes_final_2 = np.empty([0, 5])
            track_feature_2 = []
            for n, track in enumerate(tracker_2.tracks):
                if (not track.is_confirmed()) or track.time_since_update > 1:
                    continue
                box = np.reshape(np.int32(track.to_tlbr()), [1, 4])
                box_id = np.array([[track.confirm_id]])
                box = np.append(box, box_id, 1)
                boxes_final_2 = np.append(boxes_final_2, box, 0)
                track_feature_2.append(track.last_feature)
            boxes_buf_2.append(boxes_final_2)
            track_buf_feature_2.append(track_feature_2)
            # interpolate track1& track2 boxes
            if i > 0 and frame_interval > 0:
                for k in range(frame_interval):
                    boxes_buf_1[k + 1] = np.empty([0, boxes_buf_1[0].shape[1]])
                    boxes_buf_2[k + 1] = np.empty([0, boxes_buf_2[0].shape[1]])
                for ii in range(boxes_buf_1[0].shape[0]):
                    box_id = boxes_buf_1[0][ii, -1]
                    box1 = boxes_buf_1[0][ii, :-1]
                    ids = np.where(boxes_buf_1[-1][:, -1] == box_id)
                    box2 = boxes_buf_1[-1][ids]

                    if box2.shape[0] != 0:
                        inter = box_linear_interpolation(box1, box2[0, :-1], frame_interval)
                        for m in range(len(inter)):
                            inter[m] = np.append(inter[m], box_id)
                            boxes_buf_1[m + 1] = np.append(boxes_buf_1[m + 1], np.reshape(inter[m], [1, 5]), 0)
                    else:
                        box1 = np.append(box1, box_id)
                        for m in range(frame_interval):
                            boxes_buf_1[m + 1] = np.append(boxes_buf_1[m + 1], np.reshape(box1, [1, 5]), 0)

                for ii in range(boxes_buf_2[0].shape[0]):
                    box_id = boxes_buf_2[0][ii, -1]
                    box1 = boxes_buf_2[0][ii, :-1]
                    ids = np.where(boxes_buf_2[-1][:, -1] == box_id)
                    box2 = boxes_buf_2[-1][ids]

                    if box2.shape[0] != 0:
                        inter = box_linear_interpolation(box1, box2[0, :-1], frame_interval)
                        for m in range(len(inter)):
                            inter[m] = np.append(inter[m], box_id)
                            boxes_buf_2[m + 1] = np.append(boxes_buf_2[m + 1], np.reshape(inter[m], [1, 5]), 0)
                    else:
                        box1 = np.append(box1, box_id)
                        for m in range(frame_interval):
                            boxes_buf_2[m + 1] = np.append(boxes_buf_2[m + 1], np.reshape(box1, [1, 5]), 0)

        else:
            boxes_buf_1.append(boxes_buf_1[-1])
            detections_buf_1.append(detections_buf_1[-1])
            boxes_buf_2.append(boxes_buf_2[-1])
            detections_buf_2.append(detections_buf_2[-1])
            track_buf_feature_1.append(track_buf_feature_1[-1])
            track_buf_feature_2.append(track_buf_feature_2[-1])

        ###############################################################################
        #                          mix two outputs(every 24 fp)
        ###############################################################################
        if len(frame_buf_1) >= frame_num_thd & len(frame_buf_2) >= frame_num_thd:
            if i % mix_skip == 0:
                time_name = time.time()
                time_name = int(round(time_name * 1000))
                # redis_list_1 = []
                # redis_list_2 = []
                redis_list = []
                track_boxes_1 = boxes_buf_1[0]
                track_boxes_2 = boxes_buf_2[0]
                boxes1_xy = np.zeros((track_boxes_1.shape[0], 2), dtype=np.float32)
                boxes2_xy = np.zeros((track_boxes_2.shape[0], 2), dtype=np.float32)
                for mm in range(track_boxes_1.shape[0]):
                    bbox_1 = track_boxes_1[mm, :]
                    boxes1_xy[mm, :] = convert_coordinate(H_mat_1, bbox_1)
                for nn in range(track_boxes_2.shape[0]):
                    bbox_2 = track_boxes_2[nn, :]
                    boxes2_xy[nn, :] = convert_coordinate(H_mat_2, bbox_2)
                features1 = []
                features2 = []
                for mm, feature in enumerate(track_buf_feature_1[0]):
                    features1.append(feature)
                for nn, feature in enumerate(track_buf_feature_2[0]):
                    features2.append(feature)
                matches, unmatched_1, unmatched_2 = combine_video(features1, features2,
                                                                  boxes1_xy, boxes2_xy,
                                                                  track_boxes_1, track_boxes_2)
                print('matches:', matches)
                print('unmatched_1', unmatched_1)
                print('unmatched_2', unmatched_2)
                for iii in range(len(matches)):
                    mm = matches[iii][0]
                    nn = matches[iii][1]
                    # cv2.putText(frame_buf_1[0], '&2_' + str(int(track_boxes_2[nn,-1])),
                    #                (int(track_boxes_1[mm,0])+80, maximum(0, int(track_boxes_1[mm,1]) -5)), 0, 5e-3 * 200,
                    #                (0, 255, 0), 2)
                    cv2.putText(frame_buf_2[0], '1_' + str(int(track_boxes_1[mm, -1])),
                                (int(track_boxes_2[nn, 0]), maximum(0, int(track_boxes_2[nn, 1]) - 5)), 0, 5e-3 * 200,
                                (0, 255, 0), 2)
                    cv2.putText(frame_buf_1[0], '1_' + str(int(track_boxes_1[mm, -1])),
                                (int(track_boxes_1[mm, 0]), maximum(0, int(track_boxes_1[mm, 1]) - 5)), 0, 5e-3 * 200,
                                (0, 255, 0), 2)
                    x1 = (boxes1_xy[mm, 0] + boxes2_xy[nn, 0]) / 2.0
                    y1 = (boxes1_xy[mm, 1] + boxes2_xy[nn, 1]) / 2.0
                    x_str = '%.2f' % x1
                    y_str = '%.2f' % y1
                    out_dict = {'x': x_str, 'y': y_str, 'id': '1_' + str(int(track_boxes_1[mm, -1])), 't': time_name}
                    redis_list.append(out_dict)
                for iii in range(len(unmatched_1)):
                    mm = unmatched_1[iii]
                    x1 = boxes1_xy[mm, 0]
                    y1 = boxes1_xy[mm, 1]
                    x_str = '%.2f' % x1
                    y_str = '%.2f' % y1
                    #                    out_dict = {'x': x_str, 'y': y_str, 'id': str(timeID + '1_' + str(int(track_boxes_1[mm, -1]))),
                    #                                't': int(round(t1 * 1000))}
                    out_dict = {'x': x_str, 'y': y_str, 'id': '1_' + str(int(track_boxes_1[mm, -1])), 't': time_name}
                    cv2.putText(frame_buf_1[0], '1_' + str(int(track_boxes_1[mm, -1])),
                                (int(track_boxes_1[mm, 0]), maximum(0, int(track_boxes_1[mm, 1]) - 5)), 0, 5e-3 * 200,
                                (0, 255, 0), 2)
                    redis_list.append(out_dict)

                for iii in range(len(unmatched_2)):
                    nn = unmatched_2[iii]
                    x1 = boxes2_xy[nn, 0]
                    y1 = boxes2_xy[nn, 1]
                    x_str = '%.2f' % x1
                    y_str = '%.2f' % y1
                    out_dict = {'x': x_str, 'y': y_str, 'id': '2_' + str(int(track_boxes_2[nn, -1])), 't': time_name}
                    redis_list.append(out_dict)
                    cv2.putText(frame_buf_2[0], '2_' + str(int(track_boxes_2[nn, -1])),
                                (int(track_boxes_2[nn, 0]), maximum(0, int(track_boxes_2[nn, 1]) - 5)), 0, 5e-3 * 200,
                                (0, 255, 0), 2)
                rds_dict = {'data': redis_list}
                #                rds_data = json.dumps(rds_dict)
                #                data = json.dumps(rds_data)
                #                print(data)

                if write_redis and (i % rds_skip_frame == 0):
                    if redis_list != []:
                        print(redis_list)
                        data = json.dumps(redis_list)
                        try:
                            rds.lpush('AI-TRACK', data)
                        except redis.exceptions.ResponseError:
                            print('redis.exceptions.ResponseError!')
                        except redis.exceptions.TimeoutError:
                            print('redis.exceptions.TimeoutError!')
                        except redis.exceptions.ConnectionError:
                            print('redis.exceptions.ConnectionError!')
                        except redis.exceptions.RedisError:
                            print('redis.exceptions.RedisError!')
                ############################ end  v2  ########################
        ###############################################################################
        #                                display
        ###############################################################################
        # show camera1
        t3 = time.time()
        if len(frame_buf_1) >= frame_num_thd:
            boxes = boxes_buf_1[0]
            # draw track boxes
            for j in range(boxes.shape[0]):
                bbox = boxes[j, :]
                cv2.rectangle(frame_buf_1[0], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (255, 255, 255), 1)
                # cv2.putText(frame_buf_1[0], '1_'+str(int(bbox[-1])), (int(bbox[0]), maximum(0,int(bbox[1])-5)), 0, 5e-3 * 200,
                #            (0, 255, 0), 2)

            # draw detection boxes
            for det in detections_buf_1[0]:
                bbox = det.to_tlbr()
                cv2.rectangle(frame_buf_1[0], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0),
                              2)
                #########################################

                # str_line = str(track.track_id)+' '+str('%.3f'%x)+ ' '+str('%.3f'%y)+' '+time_stamp+'\n'
            if show_image:
                cv2.imshow('camera1', frame_buf_1[0])
        # show camera2
        if len(frame_buf_2) >= frame_num_thd:
            boxes = boxes_buf_2[0]
            # draw track boxes
            for j in range(boxes.shape[0]):
                bbox = boxes[j, :]
                cv2.rectangle(frame_buf_2[0], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (255, 255, 255), 1)
                # cv2.putText(frame_buf_2[0], '2_'+str(int(bbox[-1])), (int(bbox[0]), maximum(0,int(bbox[1])-5)), 0, 5e-3 * 200,
                #            (0, 255, 0), 2)

            # draw detection boxes
            for det in detections_buf_2[0]:
                bbox = det.to_tlbr()
                cv2.rectangle(frame_buf_2[0], (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                              (255, 0, 0), 2)
                #########################################

                # str_line = str(track.track_id)+' '+str('%.3f'%x)+ ' '+str('%.3f'%y)+' '+time_stamp+'\n'
            if show_image:
                cv2.imshow('camera2', frame_buf_2[0])

            if writeVideo_flag:
                # save a frame1
                out_1.write(frame_buf_1[0])
                frame_index = frame_index + 1
                list_file_1.write(str(frame_index) + ' ')
                if len(boxes) != 0:
                    for ii in range(0, len(boxes)):
                        list_file_1.write(
                            str(boxes[ii][0]) + ' ' + str(boxes[ii][1]) + ' ' + str(boxes[ii][2]) + ' ' + str(
                                boxes[ii][3]) + ' ')
                list_file_1.write('\n')
                # save a frame2
                out_2.write(frame_buf_2[0])
                list_file_2.write(str(frame_index) + ' ')
                if len(boxes) != 0:
                    for ii in range(0, len(boxes)):
                        list_file_2.write(
                            str(boxes[ii][0]) + ' ' + str(boxes[ii][1]) + ' ' + str(boxes[ii][2]) + ' ' + str(
                                boxes[ii][3]) + ' ')
                list_file_2.write('\n')

            if write_image_flag and write_redis and (i % rds_skip_frame == 0):
                # Save an image
                folder_1 = os.path.exists(image_path_1)
                folder_2 = os.path.exists(image_path_2)
                if not folder_1:
                    os.makedirs(image_path_1)
                if not folder_2:
                    os.makedirs(image_path_2)
                frame_index = frame_index + 1
                # image_file_1 = image_path_1 + str(frame_index) + '.jpg'
                # image_file_2 = image_path_2 + str(frame_index) + '.jpg'
                image_file_1 = image_path_1 + str(time_name) + '.jpg'
                # image_file_2 = image_path_2 + str(time_name) + '.jpg'
                frame_rescale_1 = cv2.resize(frame_buf_1[0], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                # frame_rescale_2 = cv2.resize(frame_buf_2[0], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(image_file_1, frame_rescale_1)
                # cv2.imwrite(image_file_2, frame_rescale_2)

            # img_path_1 = {'camera_1':image_file_1}
            # img_path_2 = {'camera_2':image_file_2}
            # data_img_1 = json.dumps(img_path_1)
            # data_img_2 = json.dumps(img_path_2)
            # print(data_img)
            # try:
            #     walking_route_image_01.lpush('walking_route_image_01', image_file_1)
            #     walking_route_image_02.lpush('walking_route_image_02', image_file_2)
            # except redis.exceptions.ResponseError:
            #     print('redis.exceptions.ResponseError!')
            # except redis.exceptions.TimeoutError:
            #     print('redis.exceptions.TimeoutError!')
            # except redis.exceptions.ConnectionError:
            #     print('redis.exceptions.ConnectionError!')
            # except redis.exceptions.RedisError:
            #     print('redis.exceptions.RedisError!')

            # delete 1st frame in buffer
            del frame_buf_1[0]
            del boxes_buf_1[0]
            del detections_buf_1[0]
            del frame_buf_2[0]
            del boxes_buf_2[0]
            del detections_buf_2[0]
            del track_buf_feature_1[0]
            del track_buf_feature_2[0]

        if show_image:
            cv2.putText(frame_buf_1[0], str(i), (900, 550), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            cv2.putText(frame_buf_2[0], str(i), (900, 550), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        i = i + 1

        if show_image:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if cv2.waitKey(1) & 0xFF == ord(' '):#  debug can delete finally
            #    cv2.waitKey(3000)
            #    cv2.waitKey(0)

        if i > 0:
            process_time.append(time.time() - t1)
            process_time_len = len(process_time)
            if process_time_len > 12:
                process_time = process_time[process_time_len - 12:process_time_len]
            print("show time=%f", time.time() - t3)
            print("time=%f, fps= %f" % (i / 25, len(process_time) / sum(process_time)))

    video_capture_1.release()
    video_capture_2.release()
    if writeVideo_flag:
        out_1.release()
        list_file_1.close()
        out_2.release()
        list_file_2.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


