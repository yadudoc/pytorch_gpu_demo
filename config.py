from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname


# Configure options here:
NODES_PER_JOB = 2
GPUS_PER_NODE = 4 
GPUS_PER_WORKER = 1

# Do not modify:
TOTAL_WORKERS = int((NODES_PER_JOB*GPUS_PER_NODE)/GPUS_PER_WORKER)
WORKERS_PER_NODE = int(GPUS_PER_NODE / GPUS_PER_WORKER)

config = Config(
    executors=[
        HighThroughputExecutor(
            label="fe.cs.uchicago",
            address=address_by_hostname(),
            max_workers=1,  # Sets #workers per manager.
            provider=SlurmProvider(
                channel=LocalChannel(),
                nodes_per_block=NODES_PER_JOB,
                init_blocks=1,
                partition='geforce',
                scheduler_options=f'#SBATCH --gres=gpu:{GPUS_PER_NODE}',
                # Launch 4 managers per node, each bound to 1 GPU
                # This is a hack. We use hostname ; to terminate the srun command, and start our own
                # DO NOT MODIFY unless you know what you are doing.
                launcher=SrunLauncher(overrides=(f'hostname; srun --ntasks={TOTAL_WORKERS} '
                                                 f'--ntasks-per-node={WORKERS_PER_NODE} '
                                                 f'--gpus-per-task={GPUS_PER_WORKER} '
                                                 f'--gpu-bind=closest')
                ),
                walltime='01:00:00',
            ),
        )
    ],
)
