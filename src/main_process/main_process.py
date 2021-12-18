import os
import sys
from typing import Optional, Tuple, List, Dict

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
from cassis_utility.loading_utility import load_cas_from_xmi_dir, \
    load_cas_from_dir, \
    find_paths_in_dir, \
    load_typesystem_from_path, \
    load_cas_from_path
from src.main_process.bucket_funcs import bucket_hansard, \
    bucket_dta, \
    bucket_bundestag, \
    bucket_coah
from src.main_process.measure_funcs import
import cassis
from tqdm import tqdm
from datetime import datetime, timedelta


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))



def process_dir_of_xmi(dir_path: str, corpus_ident: str, verbose: bool) -> Dict[str, List[Tuple[List[Tuple[int, int, float, float]], str]]]:
    """
    Paths to the Corporas:

                            DTA:     - /vol/s5935481/eger_resources/DTA                       -- warogast
                            COAH:    - /resources/corpora/COHA/texts_clean_xmi_ttlab          -- network
                                     - /vol/s5935481/eger_resources/COAH                      -- warogast
                            BT:      - /resources/corpora/Bundestag/outT2W/                   -- network
                            Hansard: - /resources/corpora/hansard_corpus/hansard_xmi_v2_ttlab -- network


    :param dir_path:
    :param corpus_ident:
    :param verbose:
    :return:
    """
    # ==== dict for using correct function ====
    bucket_determiner = {
                         "Bundestag": bucket_bundestag,
                         "COAH": bucket_coah,
                         "DTA": bucket_dta,
                         "Hansard": bucket_hansard
                         }

    # ==== getting path of typesystem for loading the cas-objects from xmi files ====
    typesystem_path = os.path.join(ROOT_DIR, "TypeSystem.xml")

    # loading cas-objects and their file-paths ====
    file_paths, typesystem_path_new = find_paths_in_dir(dir_path=dir_path)
    if typesystem_path_new is not None:
        typesystem_path = typesystem_path_new

    # ==== loading typesystem ====
    typesystem = load_typesystem_from_path(typesystem_path)

    # ==== hiding p-bar if verbose is off ====
    if verbose:
        bar = tqdm(total=len(file_paths), desc="Determine buckets for all cas-objects", position=0)
    else:
        bar = tqdm(total=len(file_paths), desc="Determine buckets for all cas-objects", position=0, disable=True)

    # ==== starting to put the single cas-objects in buckets, they get combined in a tuple with their file-path ====
    buckets = dict()
    for i in range(0, len(file_paths)):
        # ==== loading cas ====
        cas = load_cas_from_path(filepath=file_paths[i], typesystem=typesystem)

        # ==== determine year (bucket-id) ====
        year = bucket_determiner[corpus_ident]((cas, file_paths[i]))

        # ==== Performing measures on cas-object ====
        # TODO: Should be not sent based, so reeturn type of this whole function will change
        result = measurements_for_cas_sent_based(cas)

        # ==== pushing result into buckets they belong ====
        if year in buckets:
            buckets[year].append((result, file_paths[i]))
        else:
            buckets[year] = [(result, file_paths[i])]
        bar.update(1)

    return buckets



process_dir_of_xmi("/resources/corpora/COHA/texts_clean_xmi_ttlab/text_1810s_kso", "COAH", True)

#load_cas_from_corpus("/resources/corpora/paraliamentary_german/xmi_ttlab", "", True)
