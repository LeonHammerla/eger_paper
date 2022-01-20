import math
import pathlib
import os
import shutil
import sys
from typing import Optional, Tuple, List, Dict, Union
from tqdm import tqdm
import tikzplotlib
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib
from statsmodels.tsa.stattools import acf, pacf, pacf_yw
import matplotlib.pyplot as plt
from deprecated import deprecated
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

from src.main_process.main_process import load_dicts, combine_result_dicts
from src.main_process.saving_plotting_utility import MAPPING_SENT

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


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


@deprecated(reason="Function does not work and is probably not needed (would need some work)")
def auto_correlate(timeseries_data: np.ndarray,
                   nlags: int) -> np.ndarray:
    """
    Function for calculating autocorrelation.
    :param nlags:
    :param timeseries_data:
    :return:
    """
    autocorrelation = []

    for shift in range(nlags):
        correlation = np.corrcoef(timeseries_data[:-shift], timeseries_data[shift:])[0, 1]
        autocorrelation.append(correlation)

    return np.array(autocorrelation)


@deprecated(reason="Function does not work and is probably not needed (would need some work)")
def partial_auto_correlate(timeseries_data: np.ndarray,
                           nlags: int):
    """
    Function for calculating the partial autocorrelation.
    :param timeseries_data:
    :param nlags:
    :return:
    """
    # partial autocorrelation
    pac = []

    # Start by treating the data as residuals:
    # left over errors that you haven't been
    # able to fit yet.
    residuals = timeseries_data
    for shift in range(nlags):
        correlation = np.corrcoef(
            timeseries_data[:-shift], residuals[shift:])[0, 1]
        pac.append(correlation)

        # Fit the new day's data and find the residuals.
        slope, intercept = np.polyfit(timeseries_data[:-shift], residuals[shift:], 1)
        estimate = intercept + slope * timeseries_data[:-shift]
        # update residuals
        residuals[shift:] = residuals[shift:] - estimate

    return np.array(pac)


def combining_results(results: List[tuple], tuple_length: int) -> Optional[tuple]:
    """
    Function creates mean result tuple for a list of tuples.
    :param tuple_length:
    :param results:
    :return:
    """
    res_length = len(results)
    if len(results) > 0:
        mean_result_tuple = [0 for i in range(0, tuple_length)]
        for res in results:
            for idx in range(0, tuple_length):
                mean_result_tuple[idx] += res[idx]
        mean_result_tuple = [measure / res_length for measure in mean_result_tuple]
        return tuple(mean_result_tuple)
    else:
        return None



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


def plot_stats_manually(stats_dict: Dict[str, Dict],
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
    texts = [fr'{name}={stats_dict["single"][name]}' for name in stats_dict["single"].keys()]
    text_str = '\n'.join(texts)
    # --> writing them:
    with open(os.path.join(stats_path, "single_value_measurements.txt"), "w") as f:
        f.write(text_str)

    # ==== Plotting multi-value measurements and saving to a pdf and .tex files ====
    # --> pdf:
    with PdfPages(os.path.join(stats_path, "multi_value_measurements.pdf")) as pdf:
        for multi_measure in stats_dict["funcs"]:
            fig = make_fig(x=timeslices, y=stats_dict["funcs"][multi_measure], measurement_name=multi_measure)
            pdf.savefig(fig)
            tikzplotlib.save(filepath=os.path.join(stats_path, f"{multi_measure}.tex"),
                             extra_axis_parameters=["font={\\fontsize{3}{12}\selectfont}"], figure=fig)


def plot_stats_automatically(stats_dict: Dict[str, Dict],
                             stats_path: str):

    """
    Savers pre Existing figures.
    :param stats_dict:
    :param stats_path:
    :return:
    """
    # ==== Getting text string for writing single values to a .txt file ====
    texts = [fr'{name}={stats_dict["single"][name]}' for name in stats_dict["single"].keys()]
    text_str = '\n'.join(texts)
    # --> writing them:
    with open(os.path.join(stats_path, "single_value_measurements.txt"), "w") as f:
        f.write(text_str)

    # ==== Plotting multi-value measurements and saving to a pdf and .tex files ====
    # --> pdf:
    with PdfPages(os.path.join(stats_path, "multi_value_measurements.pdf")) as pdf:
        for multi_measure in stats_dict["funcs"]:
            fig = stats_dict["funcs"][multi_measure]
            pdf.savefig(fig)
            try:
                tikzplotlib.save(filepath=os.path.join(stats_path, f"{multi_measure}.tex"),
                                 extra_axis_parameters=["font={\\fontsize{3}{12}\selectfont}"], figure=fig)
            except:
                print(os.path.join(stats_path, f"{multi_measure}.tex"))

def calculate_statistics(corpus_ident: str,
                         nlags: Optional[int] = None,
                         verbose: bool = True,
                         size: int = 4,
                         plot_mechanic: str = "automatically"):
    """
    Function for Calculating statistics for a given Corpus.

    - corpus_ident: Name of Corpus (DTA, Hansard, COAH etc.)
    - nlags: number of lags that should be used for lagged statistic calculations like acf or pacf
             if None max is used
    - verbose: show progress bar
    - size: font size of plots
    - plot_mechanic: want to use built-in statsmodels plot functions or plain matplotlib figure (automatically for built-in,
                     manually else)
    :param corpus_ident:
    :param nlags:
    :param verbose:
    :param size:
    :param plot_mechanic:
    :return:
    """

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

    # ==== PBar, if verbose ====
    if verbose:
        pbar = tqdm(total=len(result_dirs), desc="Calculating Statistics for all result-dirs")
    else:
        pbar = tqdm(total=len(result_dirs), desc="Calculating Statistics for all result-dirs", disable=True)

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
                if mean_result_tuple is None:
                    pass
                else:
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

                if plot_mechanic == "manually":
                    # --> acf:
                    try:
                        acf_function = acf(x=timeseries, nlags=len(timeseries) - 1 if nlags is None else nlags)
                    except:
                        #print(timeseries)
                        acf_function = np.array([0 for i in range(0, len(timeseries) if nlags is None else nlags)])
                    # --> pacf:
                    try:
                        pacf_function = pacf_yw(x=timeseries, nlags=len(timeseries) - 1 if nlags is None else nlags, method="mle")
                    except:
                        #print(timeseries)
                        pacf_function = np.array([0 for i in range(0, len(timeseries) if nlags is None else nlags)])
                    # --> corr:
                    corr = correlation(timeseries=timeseries)
                    stats = {"single": {"cor": corr}, "funcs": {"acf": acf_function, "pacf": pacf_function}}
                    # --> plotting and result saving:
                    plot_stats_manually(stats_dict=stats,
                                        stats_path=measure_path,
                                        timeslices=np.arange(len(timeseries) if nlags is None else nlags),
                                        size=size)
                else:
                    # ==== Setting some plot params ====
                    params = {'legend.fontsize': 'large',
                              'axes.labelsize': size,
                              'axes.titlesize': size,
                              'xtick.labelsize': size,
                              'ytick.labelsize': size}
                    plt.rcParams.update(params)

                    # --> acf:
                    try:
                        acf_fig = plot_acf(x=timeseries, lags=len(timeseries) - 1 if nlags is None else nlags)
                    except:
                        # print(timeseries)
                        acf_fig = make_fig(x=np.arange(len(timeseries) if nlags is None else nlags),
                                           y=np.array([0 for i in range(0, len(timeseries) if nlags is None else nlags)]),
                                           measurement_name="acf")
                    # --> pacf:
                    try:
                        pacf_fig = plot_pacf(x=timeseries, lags=math.floor(len(timeseries)/2) -1 if nlags is None else nlags)
                        pacf_ful_lag_yw = make_fig(x=np.arange(len(timeseries) if nlags is None else nlags),
                                                   y=pacf_yw(x=timeseries, nlags=len(timeseries) - 1 if nlags is None else nlags, method="mle"),
                                                   measurement_name="pacf_ful_lag_yw")
                    except:
                        # print(timeseries)
                        pacf_fig = make_fig(x=np.arange(math.floor(len(timeseries)/2) if nlags is None else nlags),
                                            y=np.array([0 for i in range(0, math.floor(len(timeseries)/2) if nlags is None else nlags)]),
                                            measurement_name="pacf")
                        pacf_ful_lag_yw = make_fig(x=np.arange(len(timeseries) if nlags is None else nlags),
                                                   y=np.array([0 for i in range(0, len(timeseries) if nlags is None else nlags)]),
                                                   measurement_name="pacf_ful_lag_yw")
                    # --> corr:
                    corr = correlation(timeseries=timeseries)
                    stats = {"single": {"cor": corr}, "funcs": {"acf": acf_fig, "pacf": pacf_fig, "pacf_ful_lag_yw":pacf_ful_lag_yw}}
                    # --> plotting and result saving:
                    plot_stats_automatically(stats_dict=stats,
                                             stats_path=measure_path)

            pbar.update(1)


calculate_statistics("DTA")
calculate_statistics("COAH")