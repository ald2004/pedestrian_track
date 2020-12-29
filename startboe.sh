#!/bin/sh 
#export PATH=.:$PATH:/usr/local/nginx/sbin
#export ESW_PATH=/opt/pedestrian_track/api
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/pint/lib/x264/lib
mkfifo /dev/shm/rtmp_q || true
mkfifo /dev/shm/img_q || true
#nohup /usr/local/bin/python boe_merge_demo.py > /dev/shm/boe.log 2>&1 &
#nohup /usr/local/bin/python boe_merge_demo.py > /dev/null 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -i /dev/shm/rtmp_q -an -s 320*240  -f flv rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi > /dev/null 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -r 12 -i /dev/shm/rtmp_q -an -s 640*480  -f flv -r 12 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi > /dev/null 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -r 19 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 19 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi > /dev/null 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -r 7 -i /dev/shm/rtmp_q -an -s 320*240  -f flv -r 7 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi &
#nohup /usr/local/bin/python npz_to_db.py >/dev/null 2>&1 &
#nohup /usr/local/bin/python npz_to_db.py &
nohup  /usr/local/bin/python flask_server.py 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -r 13 -i /dev/shm/rtmp_q -an -s 570*320 -q:v 3 -f flv -r 13 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi > /dev/null 2>&1 &
#nohup ffmpeg -hwaccel_device /dev/dri/card0 -r 15 -i /dev/shm/rtmp_q -an -s 320*240 -q:v 3 -f flv -r 15 rtmp://192.168.8.121/live/bbb -c:v hevc_vaapi > /dev/null 2>&1 &
nohup /usr/local/bin/python boe_merge_demo.py 2>&1 &
