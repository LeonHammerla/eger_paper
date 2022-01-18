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
from src.main_process.saving_plotting_utility import make_timeslices_txts
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
                       n_procs: int) -> Tuple[Tuple[List[dict], List[dict]], dict]:
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
    # --> doc-based:
    data_dir_doc_10 = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "sent_length_02-09", "results")
    data_dir_doc_20 = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "sent_length_10-19", "results")
    data_dir_doc_30 = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "sent_length_20-29", "results")
    data_dir_doc_40 = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "sent_length_30-39", "results")
    data_dir_doc_xx = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "sent_length_40+", "results")
    # --> sent-based:
    data_dir_sent_10 = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "sent_length_02-09", "results")
    data_dir_sent_20 = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "sent_length_10-19", "results")
    data_dir_sent_30 = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "sent_length_20-29", "results")
    data_dir_sent_40 = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "sent_length_30-39", "results")
    data_dir_sent_xx = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "sent_length_40+", "results")
    # --> timeslices_safe:
    time_slices_safe = os.path.join(ROOT_DIR, "data", corpus_ident, "timeslices_safe")
    # --> data_dirs:
    data_dir_dict = {"sent": [data_dir_sent_10, data_dir_sent_20, data_dir_sent_30, data_dir_sent_40, data_dir_sent_xx],
                     "doc": [data_dir_doc_10, data_dir_doc_20, data_dir_doc_30, data_dir_doc_40, data_dir_doc_xx]}

    try:
        # --> doc-based:
        shutil.rmtree(data_dir_doc_10)
        shutil.rmtree(data_dir_doc_20)
        shutil.rmtree(data_dir_doc_30)
        shutil.rmtree(data_dir_doc_40)
        shutil.rmtree(data_dir_doc_xx)
        # --> sent-based:
        shutil.rmtree(data_dir_sent_10)
        shutil.rmtree(data_dir_sent_20)
        shutil.rmtree(data_dir_sent_30)
        shutil.rmtree(data_dir_sent_40)
        shutil.rmtree(data_dir_sent_xx)
        # --> timeslices_safe:
        shutil.rmtree(time_slices_safe)
    except:
        pass
    # --> doc-based:
    pathlib.Path(data_dir_doc_10).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_doc_20).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_doc_30).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_doc_40).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_doc_xx).mkdir(parents=True, exist_ok=True)
    # --> sent-based:
    pathlib.Path(data_dir_sent_10).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_sent_20).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_sent_30).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_sent_40).mkdir(parents=True, exist_ok=True)
    pathlib.Path(data_dir_sent_xx).mkdir(parents=True, exist_ok=True)
    # --> timeslices_safe:
    pathlib.Path(time_slices_safe).mkdir(parents=True, exist_ok=True)

    # ==== getting path of typesystem for loading the cas-objects from xmi files ====
    typesystem_path = os.path.join(ROOT_DIR, "TypeSystem.xml")

    # loading cas-objects and their file-paths ====
    file_paths, typesystem_path_new = find_paths_in_dir(dir_path=dir_path)
    if typesystem_path_new is not None:
        typesystem_path = typesystem_path_new

    # ==== Processing cas-objects (single or multi process) ====
    if n_procs <= 1:
        buckets_result, final_paths_result = process_list_of_cas_paths(cas_paths=file_paths,
                                                                  typesystem=typesystem_path,
                                                                  verbose=verbose,
                                                                  corpus_ident=corpus_ident)
        sent_based_combined_results, doc_based_combined_results = buckets_result
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
                            data_dirs=data_dir_dict)

        # ==== Declaring Pool and map processing function on to cas-obj ====
        with Pool(n_procs) as pool:
            if verbose:
                pbar = tqdm(desc=f"Processing Cas-Objects from {corpus_ident}", total=len(path_pos_chunks))
                for _ in pool.imap_unordered(part_func, path_pos_chunks):
                    pbar.update(1)
            else:
                pool.imap_unordered(part_func, path_pos_chunks)

        # ==== Loading results for sent and doc based calculations ====
        n_dicts = len(path_pos_chunks)
        # --> sent-based:
        result_sent_10 = load_dicts(dir_path=data_dir_sent_10, verbose=verbose, n_dicts=n_dicts)
        result_sent_20 = load_dicts(dir_path=data_dir_sent_20, verbose=verbose, n_dicts=n_dicts)
        result_sent_30 = load_dicts(dir_path=data_dir_sent_30, verbose=verbose, n_dicts=n_dicts)
        result_sent_40 = load_dicts(dir_path=data_dir_sent_40, verbose=verbose, n_dicts=n_dicts)
        result_sent_xx = load_dicts(dir_path=data_dir_sent_xx, verbose=verbose, n_dicts=n_dicts)
        # --> doc-based:
        result_doc_10 = load_dicts(dir_path=data_dir_doc_10, verbose=verbose, n_dicts=n_dicts)
        result_doc_20 = load_dicts(dir_path=data_dir_doc_20, verbose=verbose, n_dicts=n_dicts)
        result_doc_30 = load_dicts(dir_path=data_dir_doc_30, verbose=verbose, n_dicts=n_dicts)
        result_doc_40 = load_dicts(dir_path=data_dir_doc_40, verbose=verbose, n_dicts=n_dicts)
        result_doc_xx = load_dicts(dir_path=data_dir_doc_xx, verbose=verbose, n_dicts=n_dicts)
        # --> paths:
        paths_result = load_dicts(dir_path=time_slices_safe, verbose=verbose, n_dicts=n_dicts)

        results_sents = [result_sent_10, result_sent_20, result_sent_30, result_sent_40, result_sent_xx]
        results_docs = [result_doc_10, result_doc_20, result_doc_30, result_doc_40, result_doc_xx]

        # ==== Combining results back to one result ====
        # --> sent-based:
        sent_based_combined_results = []
        for res_sent in results_sents:
            buckets_result_sent = combine_result_dicts(res_sent)
            sent_based_combined_results.append(buckets_result_sent)
        # --> doc-based:
        doc_based_combined_results = []
        for res_doc in results_docs:
            buckets_result_doc = combine_result_dicts(res_doc)
            doc_based_combined_results.append(buckets_result_doc)
        # --> paths:
        final_paths_result = combine_result_dicts(paths_result)

    # ==== Plotting results ====
    # --> paths:
    make_timeslices_txts(paths_dict=final_paths_result,
                         corpus_ident=corpus_ident)
    # -> sent-based
    for idx in range(0, 5):
        plotting_results(result_bucket=sent_based_combined_results[idx],
                         res_type="sent",
                         verbose=verbose,
                         data_dir=data_dir_dict["sent"][idx])
    # -> doc-based
    for idx in range(0, 5):
        plotting_results(result_bucket=doc_based_combined_results[idx],
                         res_type="doc",
                         verbose=verbose,
                         data_dir=data_dir_dict["doc"][idx])

    return (sent_based_combined_results, doc_based_combined_results), final_paths_result


def process_list_of_cas_paths(cas_paths: Union[List[str], Tuple[List[str], int]],
                              typesystem: str,
                              verbose: bool,
                              corpus_ident: str,
                              data_dirs: Optional[Dict[str, List[str]]] = None) -> Optional[Tuple[Tuple[List[Dict[Any, Any]], List[Dict[Any, Any]]], Dict[str, List[str]]]]:
    """
    Function takes in a list of cas-file-paths and processes them sent or doc based.
    :param data_dirs:
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
        time_slices_safe = os.path.join(ROOT_DIR, "data", corpus_ident, "timeslices_safe", f"path_{pos}.pickle")
        #filepath1_sent = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "results", f"res_{pos}.pickle")
        #filepath2_sent = os.path.join(ROOT_DIR, "data", corpus_ident, "sent", "results", f"path_{pos}.pickle")

        #filepath1_doc = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "results", f"res_{pos}.pickle")
        #filepath2_doc = os.path.join(ROOT_DIR, "data", corpus_ident, "doc", "results", f"path_{pos}.pickle")
        verbose = False
    else:
        if verbose:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0)
        else:
            pbar = tqdm(total=len(cas_paths), desc="Processing List of Cas", leave=True, position=0, disable=True)

    # ==== Results are saved in buckets determined by their documents creation year ====
    buckets_result_doc = [dict() for y in range(0, 5)]
    buckets_result_sent = [dict() for y in range(0, 5)]
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
        result_sent_bucket_wise = [[], [], [], [], []]
        if result_sent:
            for sentence_result in result_sent:
                sent_length = sentence_result[3]
                bucket_idx = math.floor(sent_length / 10)
                if bucket_idx > 4:
                    bucket_idx = 4
                result_sent_bucket_wise[bucket_idx].append(sentence_result)

            for result_sent_bucket_wise_idx in range(0, 5):
                # ==== pushing result into buckets they belong ====
                # --> sent-based:
                if year in buckets_result_sent[result_sent_bucket_wise_idx]:
                    buckets_result_sent[result_sent_bucket_wise_idx][year].extend(result_sent_bucket_wise[result_sent_bucket_wise_idx])
                else:
                    buckets_result_sent[result_sent_bucket_wise_idx][year] = result_sent_bucket_wise[result_sent_bucket_wise_idx]
                # --> doc-based:
                result_doc = doc_based_measurements_for_cas(result_sent_bucket_wise[result_sent_bucket_wise_idx])

                if result_doc is not None:
                    if year in buckets_result_doc[result_sent_bucket_wise_idx]:
                        buckets_result_doc[result_sent_bucket_wise_idx][year].append(result_doc)
                    else:
                        buckets_result_doc[result_sent_bucket_wise_idx][year] = [result_doc]

        # ==== year-filepath-buckets ====
        if year in buckets_paths:
            buckets_paths[year].append(cas_paths[i])
        else:
            buckets_paths[year] = [cas_paths[i]]

        if verbose:
            pbar.update(1)

    if used_by_mp:
        # ==== Saving as pickle ====
        # --> file-paths:
        with open(time_slices_safe, "wb") as f:
            pickle.dump(obj=buckets_paths, file=f)
        # --> sent-based
        for idx in range(0, 5):
            with open(os.path.join(data_dirs["sent"][idx], f"res_{pos}.pickle"), "wb") as f:
                pickle.dump(obj=buckets_result_sent[idx], file=f)
        # --> doc-based
        for idx in range(0, 5):
            with open(os.path.join(data_dirs["doc"][idx], f"res_{pos}.pickle"), "wb") as f:
                pickle.dump(obj=buckets_result_doc[idx], file=f)
        return
    else:
        return (buckets_result_sent, buckets_result_doc), buckets_paths


def combine_result_dicts(result: List[dict]) -> dict:
    """
    Function combines result dicts from multiprocessing mapping result to one result tuple.
    :param result:
    :return:
    """

    buckets_result = dict()
    for i in range(0, len(result)):
        # ==== filling result ====
        for j in result[i]:
            if j in buckets_result:
                buckets_result[j].extend(result[i][j])
            else:
                buckets_result[j] = result[i][j]

    return buckets_result



def load_dicts(dir_path: str,
               verbose: bool,
               n_dicts: Optional[int] = None) -> List[dict]:
    """
    Function loads saved dicts.
    :param n_dicts:
    :param verbose:
    :param dir_path:
    :return:
    """

    # ==== Finding all result-pickle filepaths in directory ====
    filepaths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

    # ==== Sort them, so they can be easily loaded ====
    filepaths.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].rstrip(".pickle")))

    # ==== Test if all are present ====
    if n_dicts is not None:
        assert len(filepaths) == n_dicts, "some result pickles are missing?!"

    # ==== Loading them all ====
    result_dicts = []
    if verbose:
        for i in tqdm(range(0, len(filepaths)), desc=f"loading dicts"):
            with open(filepaths[i], "rb") as f:
                res_buckets = pickle.load(f)
            result_dicts.append(res_buckets)
    else:
        for i in range(0, len(filepaths)):
            with open(filepaths[i], "rb") as f:
                res_buckets = pickle.load(f)
            result_dicts.append(res_buckets)

    return result_dicts

"""
def plot_pickle_result_dir(corpus_ident: str,
                           verbose: bool,
                           return_type: str = "sent",
                           fontsize: float = 4) -> Tuple[dict, dict]:
   


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

"""

if __name__ == '__main__':

    # COAH
    res = process_dir_of_xmi(dir_path="/vol/s5935481/eger_resources/COAH/texts_clean_xmi_ttlab",
                             corpus_ident="COAH",
                             verbose=True,
                             n_procs=28)
    


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
                             n_procs=28)

    """

    """
    # Bundestag
    res = process_dir_of_xmi(dir_path="/resources/corpora/Bundestag/outT2W",
                             corpus_ident="Bundestag",
                             verbose=True,
                             n_procs=28)
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