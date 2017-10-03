import os
import platform
import ctypes.util
from subprocess import Popen, PIPE

from .env import check_env_flag

WINDOWS_HOME = 'C:/Program Files/NVIDIA Corporation/NvToolsExt'

if check_env_flag('NO_CUDA'):
    NVTOOLEXT_HOME = None
else:
    # We use nvcc path on Linux and cudart path on macOS
    osname = platform.system()
    if osname != 'Windows':
        NVTOOLEXT_HOME = None
    else:
        NVTOOLEXT_HOME = os.getenv('NVTOOLSEXT_PATH', WINDOWS_HOME).replace('\\', '/')
        if not os.path.exists(NVTOOLEXT_HOME):
            NVTOOLEXT_HOME = ctypes.util.find_library('nvToolsExt64_1')
            if NVTOOLEXT_HOME is not None:
                NVTOOLEXT_HOME = os.path.dirname(NVTOOLEXT_HOME)
            else:
                NVTOOLEXT_HOME = None
