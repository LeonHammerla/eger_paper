import os
import pathlib
import pickle
import shutil
import sys
from typing import Optional, Tuple, List, Dict, Union, Any
from multiprocessing import Pool
from functools import partial
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cassis_utility.loading_utility import load_cas_from_xmi_dir, \
    load_cas_from_dir, \
    find_paths_in_dir, \
    load_typesystem_from_path, \
    load_cas_from_path
from cassis_annotations.basic_utility import chunks
from src.main_process.bucket_funcs import bucket_hansard, \
    bucket_dta, \
    bucket_bundestag, \
    bucket_coah
from src.main_process.measure_funcs import sent_based_measurements_for_cas, doc_based_measurements_for_cas
from src.main_process.saving_plotting_utility import plotting_results
import cassis
from tqdm import tqdm
from datetime import datetime, timedelta
import math


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))



def process_dir_of_xmi(dir_path: str,
                       corpus_ident: str,
                       verbose: bool,
                       n_procs: int) -> Tuple[Tuple[Union[dict, Any], Union[dict, Any]], Union[dict, Any]]:
    """
    Paths to the Corporas:

                            DTA:     - /vol/s5935481/eger_resources/DTA/dta_kernkorpus_2020-07-20/ttlab_xmi     -- warogast
                            COAH:    - /resources/corpora/COHA/texts_clean_xmi_ttlab                            -- network
                                     - /vol/s5935481/eger_resources/COAH/texts_clean_xmi_ttlab                  -- warogast
                            BT:      - /resources/corpora/Bundestag/outT2W/                                     -- network
                            Hansard: - /resources/corpora/hansard_corpus/hansard_xmi_v2_ttlab                   -- network


    return_type --> "sent" for sentence based calculations or "doc" for document based.

    :param n_procs:
    :param dir_path:
    :param corpus_ident:
    :param verbose:
    :return:
    """

    # ==== Creating Dir if not existing and adding result folder ====
    data_dir_doc = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "results")
    data_dir_sent = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "results")
    try:
        shutil.rmtree(data_dir_doc)
        shutil.rmtree(data_dir_sent)
    except:
        pass
    pathlib.Path(data_dir_doc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_sent).mkdir(parents=True, exist_ok=True)

    # ==== getting path of typesystem for loading the cas-objects from xmi files ====
    typesystem_path = os.path.join(ROOT_DIR, "TypeSystem.xml")

    # loading cas-objects and their file-paths ====
    file_paths, typesystem_path_new = find_paths_in_dir(dir_path=dir_path)
    if typesystem_path_new is not None:
        typesystem_path = typesystem_path_new



    # ==== Processing cas-objects (single or multi process) ====
    if n_procs <= 1:
        buckets_result, buckets_paths = process_list_of_cas_paths(cas_paths=file_paths,
                                                                  typesystem=typesystem_path,
                                                                  verbose=verbose,
                                                                  corpus_ident=corpus_ident)
        buckets_result_sent, buckets_result_doc = buckets_result
    else:
        # ==== Making path-chunks and pbars ====
        path_chunks = list(chunks(file_paths, math.ceil(len(file_paths) / (n_procs * 10))))
        path_pos_chunks = []
        for i in range(0, len(path_chunks)):
            path_pos_chunks.append((path_chunks[i], i))

        # ==== Partial function for multiprocessing ====
        part_func = partial(process_list_of_cas_paths,
                            typesystem=typesystem_path,
                            verbose=False,
                            corpus_ident=corpus_ident)

        # ==== Declaring Pool and map processing function on to cas-obj ====
        with Pool(n_procs) as pool:
            if verbose:
                pbar = tqdm(desc=f"Processing Cas-Objects from {corpus_ident}", total=len(path_pos_chunks))
                for _ in pool.imap_unordered(part_func, path_pos_chunks):
                    pbar.update(1)
            else:
                pool.imap_unordered(part_func, path_pos_chunks)

        # ==== Loading results for sent and doc based calculations ====
        result_sent = load_dicts(dir_path=data_dir_sent, verbose=verbose)
        result_doc = load_dicts(dir_path=data_dir_doc, verbose=verbose)

        # ==== Combining results back to one result ====
        buckets_result_sent, _ = combine_result_dicts(result_sent)
        buckets_result_doc, buckets_paths = combine_result_dicts(result_doc)
        assert _ == buckets_paths, "something went wrong with declaring buckets for sent and doc based calcs"

    # ==== Plotting results ====
    # -> sent-based
    plotting_results(result_bucket=buckets_result_sent,
                     paths_dict=buckets_paths,
                     corpus_ident=corpus_ident,
                     res_type="sent",
                     verbose=verbose)
    # -> doc-based
    plotting_results(result_bucket=buckets_result_doc,
                     paths_dict=buckets_paths,
                     corpus_ident=corpus_ident,
                     res_type="doc",
                     verbose=verbose)

    return (buckets_result_sent, buckets_result_doc), buckets_paths


def process_list_of_cas_paths(cas_paths: Union[List[str], Tuple[List[str], int]],
                              typesystem: str,
                              verbose: bool,
                              corpus_ident: str) -> Optional[Tuple[Tuple[Dict[str, List[Tuple[int, int, int, float]]],
                                                                         Dict[str, List[Tuple[
                                                                             int, int, int, float, float, float, float]]]],
                                                                   Dict[str, List[str]]]]:
    """
    Function takes in a list of cas-file-paths and processes them sent or doc based.
    :param cas_paths:
    :param typesystem:
    :param verbose:
    :param corpus_ident:
    :return:
    """

    # ==== Determine later (if input is tuple), if function is used for mp ====
    used_by_mp = False

    # ==== loading typesystem ====
    typesystem = load_typesystem_from_path(typesystem)

    # ==== dict for using correct function ====
    bucket_determiner = {
        "Bundestag": bucket_bundestag,
        "COAH": bucket_coah,
        "DTA": bucket_dta,
        "Hansard": bucket_hansard
    }

    # ==== Setting pbar and cas-paths, differs wether used for mp oder sp ====
    if type(cas_paths) == tuple:
        used_by_mp = True
        pos = cas_paths[1]
        cas_paths = cas_paths[0]
        filepath1_sent = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "results", f"res_{pos}.pickle")
        filepath2_sent = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "results", f"path_{pos}.pickle")
        filepath1_doc = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "results", f"res_{pos}.pickle")
        filepath2_doc = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "results", f"path_{pos}.pickle")
        verbose = False
    else:
        if verbose:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0)
        else:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0, disable=True)

    # ==== Results are saved in buckets determined by their documents creation year ====
    buckets_result_doc = dict()
    buckets_result_sent = dict()
    buckets_paths = dict()

    # ==== Going through single casses ====
    for i in range(0, len(cas_paths)):


        # ==== loading cas ====
        cas = load_cas_from_path(filepath=cas_paths[i], typesystem=typesystem)

        # ==== determine year (bucket-id) ====
        year = bucket_determiner[corpus_ident]((cas, cas_paths[i]))

        # ==== Performing measures on cas-object (for sent and doc based calculations) ====
        # ==== Calculating result for sent based and combining them for doc based ====
        result_sent = sent_based_measurements_for_cas(cas)
        result_doc = doc_based_measurements_for_cas(result_sent)

        # ==== pushing result into buckets they belong ====
        # -> sent-based
        if year in buckets_result_sent:
            buckets_result_sent[year].extend(result_sent)
        else:
            buckets_result_sent[year] = result_sent
        # -> doc-based
        if year in buckets_result_doc:
            buckets_result_doc[year].append(result_doc)
        else:
            buckets_result_doc[year] = [result_doc]

        # ==== year-filepath-buckets ====
        if year in buckets_paths:
            buckets_paths[year].append(cas_paths[i])
        else:
            buckets_paths[year] = [cas_paths[i]]

        if verbose:
            pbar.update(1)

    if used_by_mp:
        # ==== Saving as pickle ====
        # -> sent-based
        with open(filepath1_sent, "wb") as f:
            pickle.dump(obj=buckets_result_sent, file=f)
        with open(filepath2_sent, "wb") as f:
            pickle.dump(obj=buckets_paths, file=f)
        # -> doc-based
        with open(filepath1_doc, "wb") as f:
            pickle.dump(obj=buckets_result_doc, file=f)
        with open(filepath2_doc, "wb") as f:
            pickle.dump(obj=buckets_paths, file=f)
        return
    else:
        return (buckets_result_sent, buckets_result_doc), buckets_paths


def combine_result_dicts(result: List[Tuple[dict, dict]]) -> Tuple[dict, dict]:
    """
    Function combines result dicts from multiprocessing mapping result to one result tuple.
    :param result:
    :return:
    """
    buckets_paths = dict()
    buckets_result = dict()
    for i in range(0, len(result)):
        # ==== filling result ====
        for j in result[i][0]:
            if j in buckets_result:
                buckets_result[j].extend(result[i][0][j])
            else:
                buckets_result[j] = result[i][0][j]
        # ==== filling paths ====
        for j in result[i][1]:
            if j in buckets_paths:
                buckets_paths[j].extend(result[i][1][j])
            else:
                buckets_paths[j] = result[i][1][j]
    return buckets_result, buckets_paths


def load_dicts(dir_path: str,
               verbose: bool) -> List[Tuple[dict, dict]]:
    """
    Function loads saved dicts.
    :param verbose:
    :param dir_path:
    :return:
    """

    # ==== Finding all result-pickle filepaths in directory ====
    filepaths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

    # ==== Sort them, so they can be easily loaded ====
    filepaths.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].rstrip(".pickle")) if "res_" in x else int(x.split("/")[-1].split("_")[-1].rstrip(".pickle")) + 0.5)

    # ==== Test if all are present ====
    assert len(filepaths) % 2 == 0, "some result pickles are missing?!"

    # ==== Loading them all ====
    result_tuples = []
    if verbose:
        for i in tqdm(range(0, len(filepaths), 2), desc=f"loading dicts"):
            with open(filepaths[i], "rb") as f:
                res_buckets = pickle.load(f)
            with open(filepaths[i+1], "rb") as f:
                path_buckets = pickle.load(f)
            result_tuples.append((res_buckets, path_buckets))
    else:
        for i in range(0, len(filepaths), 2):
            with open(filepaths[i], "rb") as f:
                res_buckets = pickle.load(f)
            with open(filepaths[i+1], "rb") as f:
                path_buckets = pickle.load(f)
            result_tuples.append((res_buckets, path_buckets))

    return result_tuples


def plot_pickle_result_dir(corpus_ident: str,
                           verbose: bool,
                           return_type: str = "sent",
                           fontsize: float = 4) -> Tuple[dict, dict]:
    """
    Function for plotting results.
    :param corpus_ident:
    :param verbose:
    :param return_type:
    :param fontsize:
    :return:
    """


    # ==== Loading results from data-dir of given corpus ====
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, return_type, "results")
    result = load_dicts(dir_path=data_dir, verbose=verbose)
    buckets_result, buckets_paths = combine_result_dicts(result)

    # ==== Plotting Results ====
    plotting_results(result_bucket=buckets_result,
                     paths_dict=buckets_paths,
                     corpus_ident=corpus_ident,
                     res_type=return_type,
                     verbose=verbose,
                     fontsize=fontsize)

    return buckets_result, buckets_paths



if __name__ == '__main__':
    """
    # COAH
    res = process_dir_of_xmi(dir_path="/vol/s5935481/eger_resources/COAH/texts_clean_xmi_ttlab",
                             corpus_ident="COAH",
                             verbose=True,
                             n_procs=28,
                             return_type="doc")
    """

    """
    # Hansard
    res = process_dir_of_xmi(dir_path="/resources/corpora/hansard_corpus/hansard_xmi_v2_ttlab",
                             corpus_ident="Hansard",
                             verbose=True,
                             n_procs=28)

    """

    """
    # DTA
    res = process_dir_of_xmi(dir_path="/vol/s5935481/eger_resources/DTA/dta_kernkorpus_2020-07-20/ttlab_xmi",
                             corpus_ident="DTA",
                             verbose=True,
                             n_procs=28,
                             return_type="doc")
    """


    # Bundestag
    res = process_dir_of_xmi(dir_path="/resources/corpora/Bundestag/outT2W",
                             corpus_ident="Bundestag",
                             verbose=True,
                             n_procs=28)


    # =======================
    # ==== JUST PLOTTING ====
    # =======================

    """
    # Bundestag -- just plotting
    res = plot_pickle_result_dir(corpus_ident="Bundestag",
                                 verbose=True,
                                 return_type="doc")
    """
    """
    # COAH -- just plotting
    res = plot_pickle_result_dir(corpus_ident="DTA",
                                 verbose=True,
                                 return_type="doc",
                                 fontsize=1.4)
    """
    """
    # print results
    doc_res_dict, _ = res
    for i in doc_res_dict:
        print(f"==================================={i}===================================")
        for j in doc_res_dict[i]:
            print(j)
    """