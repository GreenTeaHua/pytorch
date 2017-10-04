import os
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .env import check_env_flag

LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v8.0'


def find_nvcc(osname):
    if osname != 'Windows':
        proc = Popen(['which', 'nvcc'], stdout=PIPE, stderr=PIPE)
    else:
        proc = Popen(['where', 'nvcc.exe'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode().strip()
    if len(out) > 0:
        if osname == 'Windows':
            if out.find('\r\n') != -1:
                out = out.split('\r\n')[0]
            out = os.path.abspath(os.path.join(os.path.dirname(out), ".."))
            out = out.replace('\\', '/')
            out = str(out)
        return os.path.dirname(out)
    else:
        return None


if check_env_flag('NO_CUDA'):
    WITH_CUDA = False
    CUDA_HOME = None
else:
    osname = platform.system()
    if osname != 'Windows':
        CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
    else:
        CUDA_HOME = os.getenv('CUDA_PATH', WINDOWS_HOME).replace('\\', '/')
    if not os.path.exists(CUDA_HOME):
        # We use nvcc path on Linux and cudart path on macOS
        if osname == 'Linux' or osname == 'Windows':
            cuda_path = find_nvcc(osname)
        else:
            cudart_path = ctypes.util.find_library('cudart')
            if cudart_path is not None:
                cuda_path = os.path.dirname(cudart_path)
            else:
                cuda_path = None
        if cuda_path is not None:
            CUDA_HOME = os.path.dirname(cuda_path)
        else:
            CUDA_HOME = None
    WITH_CUDA = CUDA_HOME is not None
