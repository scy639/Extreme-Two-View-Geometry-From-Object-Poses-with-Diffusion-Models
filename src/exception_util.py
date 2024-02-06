import os
import root_config
import loguru 
import time
import traceback
from misc_util import your_datetime
def handle_exception(e ):
    print("\n-------[handle_exception]-------\n")
    print(f"{your_datetime():%Y.%m.%d-%H:%M:%S}")
    print(e)
    print(traceback.format_exc())
    loguru.logger.exception (e)
    with open(os.path.join(root_config.path_root,"error.txt"), "a") as f:
        f.write("\n" + "\n" + "\n" + "\n")
        f.write(f"{your_datetime():%Y.%m.%d-%H:%M:%S}" + "\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc() + "\n")