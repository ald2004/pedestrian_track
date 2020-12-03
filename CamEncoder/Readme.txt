platform:rk3399、usb-carmara
The USB camera data is encoded in H264 format


CamEncode.cpp
           ------#define DEVICE_FILE "/dev/video0"    //set device fd
           ------ init
                               initCam(CameraFd,DEVICE_FILE); //init camera
                               initMppEncoder();                        //init encoder  
                               startCapturing(CameraFd);           // start camera
           -------loop
                               main_loop   //loop
                               mppEncoderFramePut((uint8_t*)buffers[buf.index].start, buf.bytesused) //encoder
                               getCamFrameToEncoder()     //get camera data    
                               getH264Data()                       //get encoder h264 data                    



