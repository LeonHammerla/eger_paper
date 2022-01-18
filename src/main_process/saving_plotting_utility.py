import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import tikzplotlib
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
FLIERPROBS = dict(marker='x', markersize=0.5,
                      linestyle='none')
MAPPING_SENT = ["n_verbs0", "complexity_ratio1", "absolute_complexity2", "order_of_tree3",
                    "dependency_index4", "stratum_of_tree5", "depth_of_tree6",
                    "ratio_vertices_on_lp_from_rt7", "leaf_distance_entropy8",
                    "ratio_of_leafs_at_d1_to_rt9", "mean_dependency_distance10",
                    "dependency_distance_entropy11", "ratio_arcs_adjacent_tokens12",
                    "ratio_arcs_distances_occ_once13", "imbalance_index14", "ratio_of_leafs15",
                    "number_of_leafs16", "ratio_arcs_CJ17", "ratio_arcs_CP17", "ratio_arcs_DA17", "ratio_arcs_HD17",
                    "ratio_arcs_MO17", "ratio_arcs_NK17", "ratio_arcs_OA17", "ratio_arcs_OA217", "ratio_arcs_OC17",
                    "ratio_arcs_PD17", "ratio_arcs_RC17",
                    "ratio_arcs_SB17", "number_arcs_CJ18", "number_arcs_CP18", "number_arcs_DA18", "number_arcs_HD18",
                    "number_arcs_MO18", "number_arcs_NK18", "number_arcs_OA18", "number_arcs_OA218", "number_arcs_OC18",
                    "number_arcs_PD18", "number_arcs_RC18", "number_arcs_SB18", "width_of_tree19",
                    "lowest_lv_max_width20", "ratio_vertices_belonging_latter_level21",
                    "Hirsch_index22", "ratio_vertices_contributing_h_index23", "relative_h_index24"]


def save_file_paths_time_slices(paths_dict: Dict[str, List[str]],
                                data_dir: str):
    """
    Function Saves Time-Slices as a .txt file containing every filepath to a cas-object
    belonging to that time slice.
    :param data_dir:
    :param paths_dict:
    :return:
    """

    # ==== Writing content of each time slice to its own .txt file ====
    for time_bucket in paths_dict:
        with open(os.path.join(data_dir, f"{time_bucket}.txt"), "w") as f:
            for path in paths_dict[time_bucket]:
                f.write(path + "\n")


def make_timeslices_txts(paths_dict: dict,
                         corpus_ident: str):
    """
    Function for making folder containing all texts for each timeslice.
    :param paths_dict:
    :param corpus_ident:
    :return:
    """
    # ==== Removing dir if already exists to make a new clean one
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, "timeslices")
    try:
        shutil.rmtree(data_dir)
    except:
        pass
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

    # ==== Saving time slices -> filepaths ====
    save_file_paths_time_slices(paths_dict=paths_dict,
                                data_dir=data_dir)


def box_plot_of_result_at_idx(y_label: str,
                              result_tuple_index: int,
                              result_dict: dict,
                              fontsize: float = 4) -> matplotlib.figure.Figure:
    """
    Function for getting
    :param y_label:
    :param fontsize:
    :param res_type:
    :param result_tuple_index:
    :param result_dict:
    :return:
    """

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
    fig = df.plot(ylabel=y_label, kind="box", fontsize=fontsize, flierprops=FLIERPROBS, showfliers=True, rot=90).get_figure()
    return fig


def plotting_results(result_bucket: dict,
                     res_type: str,
                     verbose: bool,
                     data_dir: str,
                     fontsize: float = 4):
    """
    Function for plotting all results."
    :param data_dir:
    :param fontsize:
    :param result_bucket:
    :param res_type:
    :param verbose:
    :return:
    """

    # ==== Removing dir if already exists to make a new clean one ====
    tex_files_path = os.path.join("/".join(data_dir.split("/")[:-1]), "tex_files")
    try:
        os.remove(os.path.join("/".join(data_dir.split("/")[:-1]), "box_plots.pdf"))
    except:
        pass
    try:
        shutil.rmtree(tex_files_path)
    except:
        pass
    pathlib.Path(tex_files_path).mkdir(parents=True, exist_ok=True)

    # ==== Finding Tuple-Length ====
    tuple_length = 0
    for i in result_bucket:
        for j in result_bucket[i]:
            tuple_length = len(j)
            break
        break

    # ==== Creating pbar ====
    if verbose:
        pbar = tqdm(total=tuple_length, desc=f"Saving Plots: {data_dir}", leave=True, disable=False)
    else:
        pbar = tqdm(total=tuple_length, desc=f"Saving Plots: {data_dir}", leave=True, disable=True)

    # ==== Plotting all different Results ====
    mapping_doc = MAPPING_SENT
    mapping_type = {"sent": MAPPING_SENT, "doc": mapping_doc}

    with PdfPages(os.path.join("/".join(data_dir.split("/")[:-1]), "box_plots.pdf")) as pdf:
        for i in range(0, tuple_length):
            y_label = mapping_type[res_type][i]
            fig = box_plot_of_result_at_idx(y_label=y_label,
                                            result_tuple_index=i,
                                            result_dict=result_bucket,
                                            fontsize=fontsize)
            pdf.savefig(fig)
            tikzplotlib.save(filepath=os.path.join(tex_files_path, f"{y_label}.tex"), extra_axis_parameters=["font={\\fontsize{3}{12}\selectfont}"], figure=fig)

            pbar.update(1)