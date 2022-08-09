'''
This script downloads the data and uploads it to W&B
Author: Diego Iglesias
Date: 20220809
'''

import pandas
import logging
import wandb
import pathlib
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    


