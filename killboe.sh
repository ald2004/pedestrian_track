ps -ef|grep boe_merge_demo|awk '{print $2}'|xargs kill -9
ps -ef|grep npz_to_db|awk '{print $2}'|xargs kill -9
ps -ef|grep flask_server|awk '{print $2}'|xargs kill -9
ps -ef|grep ffmpeg|awk '{print $2}'|xargs kill -9
