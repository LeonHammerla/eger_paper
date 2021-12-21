import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def save_file_paths_time_slices(paths_dict: Dict[str, List[str]],
                                corpus_ident: str):
    """
    Function Saves Time-Slices as a .txt file containing every filepath to a cas-object
    belonging to that time slice.
    :param paths_dict:
    :param corpus_ident:
    :return:
    """
    # ==== Creating Path if not exists ====
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, "timeslices")
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    # ==== Writing content of each time slice to its own .txt file ====
    for time_bucket in paths_dict:
        with open(os.path.join(data_dir, f"{time_bucket}.txt"), "w") as f:
            for path in paths_dict[time_bucket]:
                f.write(path + "\n")


def box_plot_of_result_at_idx(result_tuple_index: int,
                              result_dict: dict,
                              res_type: str) -> matplotlib.figure.Figure:
    """
    Function for getting
    :param res_type:
    :param result_tuple_index:
    :param result_dict:
    :return:
    """
    # ==== Mapping Dicts for correct indices ====
    mapping_sent = ["n_toks", "n_verbs", "max_depth", "mdd"]
    mapping_doc = ["n_token", "n_verbs", "n_sents", "tok_per_sentence", "v_per_sentence", "mdd", "avg_max_depth"]
    mapping_type = {"sent": mapping_sent, "doc": mapping_doc}
    data = {}
    max_length = 0

    # ==== Going through result time-slices and picking result-type at specified index ====
    for time_slice in result_dict:
        time_slice_data = []
        for result_tuple in result_dict[time_slice]:
            time_slice_data.append(result_tuple[result_tuple_index])
        if max_length < len(time_slice_data):
            max_length = len(time_slice_data)
        data[time_slice] = time_slice_data
    for time_slice in data:
        data[time_slice] = data[time_slice] + [np.nan for i in range(0, max_length - len(data[time_slice]))]

    # ==== Constructing pandas dataframe and plt figure, sorting by year ====
    data = {k: v for k, v in sorted(data.items(), key=lambda v: int(v[0]))}
    df = pd.DataFrame(data=data)
    fig = df.plot(ylabel=mapping_type[res_type][result_tuple_index], kind="box").get_figure()
    return fig


def plotting_results(result_bucket: dict,
                     paths_dict: dict,
                     corpus_ident: str,
                     res_type: str,
                     verbose: bool):
    """
    Function for plotting all results"
    :param result_bucket:
    :param paths_dict:
    :param corpus_ident:
    :param res_type:
    :param verbose:
    :return:
    """
    # ==== Removing dir if already exists to make a new clean one
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident)
    try:
        shutil.rmtree(os.path.join(ROOT_DIR, "data", corpus_ident, "timeslices"))
    except:
        pass
    try:
        os.remove(os.path.join(ROOT_DIR, "data", corpus_ident, "box_plots.pdf"))
    except:
        pass

    # ==== Saving time slices -> filepaths ====
    save_file_paths_time_slices(paths_dict=paths_dict,
                                corpus_ident=corpus_ident)

    # ==== Finding Tuple-Length ====
    tuple_length = 0
    for i in result_bucket:
        for j in result_bucket[i]:
            tuple_length = len(j)
            break
        break

    # ==== Creating pbar ====
    if verbose:
        pbar = tqdm(total=tuple_length, desc="Saving Plots", leave=True, disable=False)
    else:
        pbar = tqdm(total=tuple_length, desc="Saving Plots", leave=True, disable=True)

    # ==== Plotting all different Results ====
    with PdfPages(os.path.join(data_dir, 'box_plots.pdf')) as pdf:
        for i in range(0, tuple_length):
            fig = box_plot_of_result_at_idx(result_tuple_index=i,
                                            result_dict=result_bucket,
                                            res_type=res_type)
            pdf.savefig(fig)
            pbar.update(1)