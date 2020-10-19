import parsl
import argparse
from config import config

@parsl.bash_app
def torch_mnist(batch_size=60, epochs=10, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    """ This bash app calls the torch_mnist.py example python script from the commandline
    with two kwarg options.
    """
    return f"""cd /home/yadunand/parsl_example
    python3 torch_mnist.py --batch-size={batch_size} --epochs={epochs}
    """

@parsl.python_app
def platinfo(sleep_dur=10):
    import platform
    import torch
    import time
    import os
    name = platform.uname().node
    dev_count = torch.cuda.device_count()
    time.sleep(sleep_dur)
    
    return f"Node:{name}, Device_count:{dev_count}, Visible_devices:{os.environ['CUDA_VISIBLE_DEVICES']}"

def test_platform():

    futures = [platinfo() for i in range(20)]
    for f in futures:
        print("Platform result :", f.result())
        futures = [platinfo() for i in range(4)]

def run_mnist(batch_range=[], epoch_range=[]):

    futures = {}
    for batch_size in batch_range:
        for epochs in epoch_range:
            futures[(batch_size, epochs)] = torch_mnist(batch_size=batch_size,
                                                        epochs=epochs)

    for f in futures:
        print(f"Result for {f}: ", futures[f].result())


if __name__ == '__main__':

    parsl.load(config)

    test_platform()
    
    run_mnist(batch_range=[60, 64],
              epoch_range=[14, 16, 18, 20])
