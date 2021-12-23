import os
import pathlib
import pickle
import shutil
import sys
from typing import Optional, Tuple, List, Dict, Union
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
                       n_procs: int,
                       return_type: str = "sent") -> Tuple[dict, dict]:
    """
    Paths to the Corporas:

                            DTA:     - /vol/s5935481/eger_resources/DTA/dta_kernkorpus_2020-07-20/ttlab_xmi     -- warogast
                            COAH:    - /resources/corpora/COHA/texts_clean_xmi_ttlab                            -- network
                                     - /vol/s5935481/eger_resources/COAH/texts_clean_xmi_ttlab                  -- warogast
                            BT:      - /resources/corpora/Bundestag/outT2W/                                     -- network
                            Hansard: - /resources/corpora/hansard_corpus/hansard_xmi_v2_ttlab                   -- network


    return_type --> "sent" for sentence based calculations or "doc" for document based.

    :param n_procs:
    :param return_type:
    :param dir_path:
    :param corpus_ident:
    :param verbose:
    :return:
    """

    # ==== Creating Dir if not existing and adding result folder ====
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, "results")
    try:
        shutil.rmtree(data_dir)
    except:
        pass
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)

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
                                                                  corpus_ident=corpus_ident,
                                                                  return_type=return_type)
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
                            corpus_ident=corpus_ident,
                            return_type=return_type)

        # ==== Declaring Pool and map processing function on to cas-obj ====
        with Pool(n_procs) as pool:
            if verbose:
                pbar = tqdm(desc=f"Processing Cas-Objects from {corpus_ident}", total=len(path_pos_chunks))
                for _ in pool.imap_unordered(part_func, path_pos_chunks):
                    pbar.update(1)
            else:
                pool.imap_unordered(part_func, path_pos_chunks)


        result = load_dicts(dir_path=data_dir, verbose=verbose)

        # ==== Combining results back to one result ====
        buckets_result, buckets_paths = combine_result_dicts(result)



    plotting_results(result_bucket=buckets_result,
                     paths_dict=buckets_paths,
                     corpus_ident=corpus_ident,
                     res_type=return_type,
                     verbose=verbose)

    return buckets_result, buckets_paths


def process_list_of_cas_paths(cas_paths: Union[List[str], Tuple[List[str], int]],
                              typesystem: str,
                              verbose: bool,
                              corpus_ident: str,
                              return_type: str) -> Optional[Tuple[dict, dict]]:
    """
    Function takes in a list of cas-file-paths and processes them sent or doc based.
    :param cas_paths:
    :param typesystem:
    :param verbose:
    :param corpus_ident:
    :param return_type:
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
        filepath1 = os.path.join(ROOT_DIR, "data", corpus_ident, "results", f"res_{pos}.pickle")
        filepath2 = os.path.join(ROOT_DIR, "data", corpus_ident, "results", f"path_{pos}.pickle")
        verbose = False
    else:
        if verbose:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0)
        else:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0, disable=True)

    # ==== Results are saved in buckets determined by their documents creation year ====
    buckets_result = dict()
    buckets_paths = dict()

    # ==== Going through single casses ====
    for i in range(0, len(cas_paths)):


        # ==== loading cas ====
        cas = load_cas_from_path(filepath=cas_paths[i], typesystem=typesystem)

        # ==== determine year (bucket-id) ====
        year = bucket_determiner[corpus_ident]((cas, cas_paths[i]))

        # ==== Performing measures on cas-object ====
        # TODO: Should be not sent based, so reeturn type of this whole function will change
        if return_type == "doc":

            # ==== Calculating result for sent based and combining them for doc based ====
            result = sent_based_measurements_for_cas(cas)
            result = doc_based_measurements_for_cas(result)

            # ==== pushing result into buckets they belong ====
            if year in buckets_result:
                buckets_result[year].append(result)
            else:
                buckets_result[year] = [result]

            if year in buckets_paths:
                buckets_paths[year].append(cas_paths[i])
            else:
                buckets_paths[year] = [cas_paths[i]]

        else:

            # ==== Calculating result for sent based ====
            result = sent_based_measurements_for_cas(cas)

            # ==== pushing result into buckets they belong ====
            if year in buckets_result:
                buckets_result[year].extend(result)
            else:
                buckets_result[year] = result

            if year in buckets_paths:
                buckets_paths[year].append(cas_paths[i])
            else:
                buckets_paths[year] = [cas_paths[i]]

        if verbose:
            pbar.update(1)

    if used_by_mp:
        with open(filepath1, "wb") as f:
            pickle.dump(obj=buckets_result, file=f)
        with open(filepath2, "wb") as f:
            pickle.dump(obj=buckets_paths, file=f)
        return
    else:
        return buckets_result, buckets_paths


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
                           return_type: str = "sent") -> Tuple[dict, dict]:


    # ==== Loading results from data-dir of given corpus ====
    data_dir = os.path.join(ROOT_DIR, "data", corpus_ident, "results")
    result = load_dicts(dir_path=data_dir, verbose=verbose)
    buckets_result, buckets_paths = combine_result_dicts(result)

    # ==== Plotting Results ====
    plotting_results(result_bucket=buckets_result,
                     paths_dict=buckets_paths,
                     corpus_ident=corpus_ident,
                     res_type=return_type,
                     verbose=verbose)

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


    # Hansard
    res = process_dir_of_xmi(dir_path="/resources/corpora/hansard_corpus/hansard_xmi_v2_ttlab",
                             corpus_ident="Hansard",
                             verbose=True,
                             n_procs=28,
                             return_type="doc")


    """
    # DTA
    res = process_dir_of_xmi(dir_path="/vol/s5935481/eger_resources/DTA/dta_kernkorpus_2020-07-20/ttlab_xmi",
                             corpus_ident="DTA",
                             verbose=True,
                             n_procs=28,
                             return_type="doc")
    """

    """
    # Bundestag
    res = process_dir_of_xmi(dir_path="/resources/corpora/Bundestag/outT2W",
                             corpus_ident="Bundestag",
                             verbose=True,
                             n_procs=28,
                             return_type="doc")
    """

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
    # print results
    doc_res_dict, _ = res
    for i in doc_res_dict:
        print(f"==================================={i}===================================")
        for j in doc_res_dict[i]:
            print(j)
    """