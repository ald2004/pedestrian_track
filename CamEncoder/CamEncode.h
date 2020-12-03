#ifndef CAMENCODE_H
#define CAMENCODE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <string.h>
#include <errno.h>
#include <rk_mpi.h>


typedef struct MPP_ENC_DATA
{
	// global flow control flag
	uint32_t frm_eos;
	uint32_t pkt_eos;
	uint32_t frame_count;
	uint64_t stream_size;

	// base flow context
	MppCtx ctx;
	MppApi *mpi;
	MppEncPrepCfg prep_cfg;
	MppEncRcCfg rc_cfg;
	MppEncCodecCfg codec_cfg;

	// input / output
	MppBuffer frm_buf;
	MppEncSeiMode sei_mode;

	uint32_t width;
	uint32_t height;
	uint32_t hor_stride;
	uint32_t ver_stride;
	MppFrameFormat fmt;
	MppCodingType type;
	uint32_t num_frames;

	// resources
	size_t frame_size;

	int32_t gop;
	int32_t fps;
	int32_t bps;

	FILE *fp_output;
}MPP_ENC_DATA;

typedef struct buffer
{
    void   *start;
    size_t  length;
}buffer;




class CamEncode
{

public:
    CamEncode();

    void start();
	enum IO_METHOD
	{
	    IO_METHOD_READ,
	    IO_METHOD_MMAP,
	    //IO_METHOD_USERPTR,
	};
    void getH264Data(void *ptr,size_t len);  //H264    data interface
	

private:
    bool mppEncoderFramePut(uint8_t *p, int size);
    int getCamFrame(int fd);
    void main_loop(int fd);
    void stopCapturing(int fd);
    int startCapturing(int fd);
    void freeBuf(void);
    int initReadBuf(unsigned int buffer_size);
    int initMmap(int fd);
    int initCam(int fd,const char* cam_dev);
    void initMppEncoder();
    void closeCamFd(int fd);
    void destroyMpp();
    int getCamFd(const char* cam_dev);
	static int xioctl(int fh, int request, void *arg);

private:
    int CameraFd;
	int CameraFps;
	int ForceFormat  = 1;
	int CamHeight    = 720;
	int CamWidth     = 1280;
	MPP_ENC_DATA mpp_enc_data;
	buffer *buffers;
	unsigned int n_buffers;
	int io_method;
	bool H264StoreFlag = false;
};


#endif

