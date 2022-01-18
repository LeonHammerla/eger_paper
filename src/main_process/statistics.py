import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(ROOT_DIR)

from src.main_process.main_process import load_dicts, combine_result_dicts
from src.main_process.saving_plotting_utility import MAPPING_SENT

def find_all_result_paths(path: str) -> [str]:
    """
    Function for finding all dirs containing pickled results in the
    complete result directory for one corpus.
    :param path:
    :return:
    """
    dir_stack = []
    result_dirs = []
    while True:
        names = [name for name in os.listdir(path)]
        for name in names:
            path_name = os.path.join(path, name)
            if os.path.isdir(path_name):
                if name == "results":
                    result_dirs.append(os.path.join(path, name))
                dir_stack.append(path_name)

        if not dir_stack:
            break
        else:
            path = dir_stack.pop()
    return result_dirs


def combining_results(results: List[tuple], tuple_length: int) -> tuple:
    """
    Function creates mean result tuple for a list of tuples.
    :param tuple_length:
    :param results:
    :return:
    """
    res_length = len(results)
    mean_result_tuple = [0 for i in range(0, tuple_length)]
    for res in results:
        for idx in range(0, tuple_length):
            mean_result_tuple[idx] += res[idx]
    mean_result_tuple = [measure / res_length for measure in mean_result_tuple]
    return tuple(mean_result_tuple)


def calculate_statistics(corpus_ident: str):

    tuple_length = len(MAPPING_SENT)
    # ==== Getting paths for pickled result dicts and loading them ====
    # --> getting paths:
    corpus_result_path = os.path.join(ROOT_DIR, "data", corpus_ident)
    result_dirs = find_all_result_paths(path=corpus_result_path)
    # --> split them into sent and doc based results:
    result_dirs_dict = {"doc":[], "sent":[]}
    for result_dir in result_dirs:
        # --> also loading them:
        res_dict = combine_result_dicts(result=load_dicts(dir_path=result_dir, verbose=False))
        result_dirs_dict[result_dir.split("/")[-3]].append((result_dir, res_dict))

    # ==== Calculating statistics ====
    for res_type in result_dirs_dict:
        for res_tuple in result_dirs_dict[res_type]:
            path, res_dict = res_tuple
            # --> making statistics folder:
            statistics_path = os.path.join("/".join(path.split("/")[:-1]), "statistics")
            try:
                shutil.rmtree(statistics_path)
            except:
                pass
            pathlib.Path(statistics_path).mkdir(parents=True, exist_ok=True)
            # --> getting timeslices with one mean_result_tuple each so basically the timeseries:
            mean_res_dict = dict()
            for timeslice in res_dict:
                mean_result_tuple = combining_results(results=res_dict[timeslice], tuple_length=tuple_length)
                mean_res_dict[timeslice] = mean_result_tuple

            for measure_idx in range(0, tuple_length):
                pathlib.Path(os.path.join(statistics_path, MAPPING_SENT[measure_idx])).mkdir(parents=True, exist_ok=True)




calculate_statistics("Hansard")