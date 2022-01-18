import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict, Union

import tikzplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib
from statsmodels.tsa.stattools import acf, pacf
import matplotlib.pyplot as plt
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


def correlation(timeseries: np.ndarray) -> float:
    """
    Function returns the correlation coefficient for a given
    timeseries.
    :param timeseries:
    :return:
    """
    n_legs = len(timeseries)
    legs = np.arange(n_legs)
    return np.corrcoef(legs, timeseries)[0, 1]


def make_fig(x: np.ndarray, y: np.ndarray, measurement_name: str) -> matplotlib.figure.Figure:
    """
    Function creates figure.
    :param x:
    :param y:
    :param measurement_name:
    :return:
    """
    fig = plt.figure()
    plt.plot(x, y)
    fig.suptitle(measurement_name + "\n", fontweight="bold")
    return fig


def plot_stats(stats_dict: Dict[str, Dict],
               stats_path: str,
               timeslices: np.ndarray,
               size: int = 4):
    """
    Function plots all stats and saves them all in one big pdf
    and every measure seperate in a .tex-file
    all single valued measurements are saved in one .txt file.
    :param stats_dict:
    :param stats_path:
    :param timeslices:
    :param size:
    :return:
    """
    # ==== Setting some plot params ====
    params = {'legend.fontsize': 'large',
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size,
              'ytick.labelsize': size}
    plt.rcParams.update(params)

    # ==== Getting text string for writing single values to a .txt file ====
    texts = [fr'$\{name}=%.2f$' % (stats_dict["single"][name],) for name in stats_dict["single"].keys()]
    text_str = '\n'.join(texts)
    # --> writing them:
    with open(os.path.join(stats_path, "single_value_measurements.txt"), "w") as f:
        f.write(text_str)

    # ==== Plotting multi-value measurements and saving to a pdf and .tex files ====
    # --> pdf:
    with PdfPages(os.path.join(stats_path, "multi_value_measurements.pdf")) as pdf:
        for multi_measure in stats_dict["multi"]:
            fig = make_fig(x=timeslices, y=stats_dict["multi"][multi_measure], measurement_name=multi_measure)
            pdf.savefig(fig)
            tikzplotlib.save(filepath=os.path.join(stats_path, f"{multi_measure}.tex"),
                             extra_axis_parameters=["font={\\fontsize{3}{12}\selectfont}"], figure=fig)


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
            # --> order by date for timeseries:
            mean_res_dict = {k: v for k, v in sorted(list(mean_res_dict.items()))}
            # --> go through every measure and do statistics:
            timeseries_measurement_data = []
            for measure_idx in range(0, tuple_length):
                measure_path = os.path.join(statistics_path, MAPPING_SENT[measure_idx])
                pathlib.Path(measure_path).mkdir(parents=True, exist_ok=True)
                timeseries = []
                timeslices = []
                for timeslice in mean_res_dict:
                    timeseries.append(mean_res_dict[timeslice][measure_idx])
                    timeslices.append(timeslice)
                timeseries_measurement_data.append(timeseries)
                timeseries = np.array(timeseries)
                timeslices = np.array(timeslices)
                # --> acf:
                acf_function = acf(x=timeseries, nlags=len(timeseries))
                # --> pacf:
                pacf_function = pacf(x=timeseries, nlags=len(timeseries))
                # --> corr:
                corr = correlation(timeseries=timeseries)
                stats = {"single": {"cor": corr}, "funcs": {"acf": acf, "pacf": pacf}}




calculate_statistics("Hansard")