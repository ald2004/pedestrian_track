from ctypes import *
import math
import random
import os

lib = CDLL(os.path.join(os.environ.get('ESW_PATH', './'), "libeswcamera.so"), RTLD_GLOBAL)


def get_ins():
    return lib.camencode_ins(None)


def start(ins):
    return lib.CallMemberTest(ins)

def testpthreads():
    return lib.go4thread(None)

lib.camencode_ins.argtypes = [c_void_p]
lib.camencode_ins.restype = c_void_p

lib.CallMemberTest.argtypes = [c_void_p]
lib.CallMemberTest.restype = c_void_p

lib.go4thread.argtypes = [c_void_p]
lib.go4thread.restype = c_void_p
# spam = get_ins()
# start(spam)
# testpthreads()

# f=open('xxx','rb')
# f.read()