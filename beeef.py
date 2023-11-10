import json
import time
from typing import Any, Dict, List, Union
import torch
import bmtrain as bmt
import os
from cpm_live.arguments import get_args

from cpm_live.models import CPMBee, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from cpm_live.utils import allgather_objects, LogManager
from cpm_live.training_tasks.bee import MixedDataset
#Li Shangfu: Chinese defence  sacked
