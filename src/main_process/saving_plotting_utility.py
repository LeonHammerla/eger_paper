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
FLIERPROBS = dict(marker='x', markersize=0.5,
                      linestyle='none')


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
                              res_type: str,
                              fontsize: float = 4) -> matplotlib.figure.Figure:
    """
    Function for getting
    :param fontsize:
    :param res_type:
    :param result_tuple_index:
    :param result_dict:
    :return:
    """
    # ==== Mapping Dicts for correct indices ====
    mapping_sent = ["n_verbs0", "complexity_ratio1", "absolute_complexity2", "order_of_tree3",
                    "dependency_index4", "stratum_of_tree5", "depth_of_tree6",
                    "ratio_vertices_on_lp_from_rt7", "leaf_distance_entropy8",
                    "ratio_of_leafs_at_d1_to_rt9", "mean dependency distance10",
                    "dependency distance entropy11", "ratio_arcs_adjacent_tokens12",
                    "ratio_arcs_distances_occ_once13", "imbalance_index14", "ratio of leafs15",
                    "number of leafs16", "ratio_arcs_CJ17", "ratio_arcs_CP17", "ratio_arcs_DA17", "ratio_arcs_HD17",
                    "ratio_arcs_MO17", "ratio_arcs_NK17", "ratio_arcs_OA17", "ratio_arcs_OA217", "ratio_arcs_OC17", "ratio_arcs_PD17", "ratio_arcs_RC17",
                    "ratio_arcs_SB17", "number_arcs_CJ18", "number_arcs_CP18", "number_arcs_DA18", "number_arcs_HD18",
                    "number_arcs_MO18", "number_arcs_NK18", "number_arcs_OA18", "number_arcs_OA218", "number_arcs_OC18",
                    "number_arcs_PD18", "number_arcs_RC18", "number_arcs_SB18", "width_of_tree19",
                    "lowest_lv_max_width20", "ratio_vertices_belonging_latter_level21",
                    "Hirsch_index22", "ratio_vertices_contributing_h_index23", "relative_h_index24"]

    mapping_doc = mapping_sent
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
    fig = df.plot(ylabel=mapping_type[res_type][result_tuple_index], kind="box", fontsize=fontsize, flierprops=FLIERPROBS, showfliers=True, rot=90).get_figure()
    return fig


def plotting_results(result_bucket: dict,
                     paths_dict: dict,
                     corpus_ident: str,
                     res_type: str,
                     verbose: bool,
                     fontsize: float = 4):
    """
    Function for plotting all results"
    :param fontsize:
    :param result_bucket:
    :param paths_dict:
    :param corpus_ident:
    :param res_type:
    :param verbose:
    :return:
    """
    # ==== Removing dir if already exists to make a new clean one
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, res_type)
    try:
        shutil.rmtree(os.path.join(ROOT_DIR, "data", corpus_ident, res_type, "timeslices"))
    except:
        pass
    try:
        os.remove(os.path.join(ROOT_DIR, "data", corpus_ident, res_type, "box_plots.pdf"))
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
                                            res_type=res_type,
                                            fontsize=fontsize)
            pdf.savefig(fig)
            pbar.update(1)