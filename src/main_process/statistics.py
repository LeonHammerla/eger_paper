import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(ROOT_DIR)

from src.main_process.