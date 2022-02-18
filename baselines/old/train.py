# Simple env test.
import logging
import os

import baselines.old.aicrowd_helper as aicrowd_helper
from baselines.old.utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, 'mod')))
from baselines.old.mod.dqn_family import main as dqn_family_main


# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
# MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondDenseVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

def main():
    """
    This function will be called for training phase.

    **IMPORTANT NOTICE**:
    The trained weights in `train/` directory of this repository were generated by `mod/dqn_family.py::main` entry point,
    not by this script.
    I've not checked if this script (`train.py`) could work on the MineRL Competition's submission system.
    (On the Round 1, participants are to submit pre-trained agents.
    You have to make your training script work on the competition submission system on the Round 2.)

    For the detail of the options of `dqn_family_main` called below, see "README#How to Train Baseline Agent on you own" section.
    """
    dqn_family_main()

    # Training 100% Completed
    aicrowd_helper.register_progress(1)


if __name__ == "__main__":
    main()
