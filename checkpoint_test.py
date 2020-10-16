import parsl
import argparse
import os
from config import config


@parsl.python_app
def torch_mnist(module_path, batch_size=60, epochs=10, checkpoint_input='', checkpoint_output=''):
    # Add the current dir to the path to make sure the torch_mnist_checkpointed module can be imported
    import sys
    sys.path.append(module_path)
    from torch_mnist_checkpointed import train_model
    import time

    start = time.time()
    train_model(batch_size=batch_size,
                epochs=epochs,
                save_model=True,
                # Checkpoint every minute so that we can recover from failures.
                checkpoint_period=1, 
                checkpoint_input=checkpoint_input,
                checkpoint_output=checkpoint_output)
    return time.time() - start
                
    
def run_mnist(batch_range=[], epoch_range=[]):

    futures = {}
    index = 0
    for batch_size in batch_range:
        for epochs in epoch_range:
            print(f"Launching mnist train with batch_size:{batch_size} and epochs:{epochs}")
            futures[(batch_size, epochs)] = torch_mnist(os.getcwd(),
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        checkpoint_input=f'mnist.{index}.pkl',
                                                        checkpoint_output=f'mnist.{index}.pkl')
            index+=1

    print("Waiting for results...")
    for f in futures:
        print(f"Result for {f}: ", futures[f].result())


if __name__ == '__main__':

    # Setting walltime to a small number to trigger failure during training
    config.executors[0].provider.walltime = '00:02:00'

    # Set retries so that each application will rerun 3 times before
    # it is deemed failed.
    config.retries = 3
    
    parsl.load(config)
    
    run_mnist(batch_range=[60, 64],
              epoch_range=[14, 16, 18, 20])
