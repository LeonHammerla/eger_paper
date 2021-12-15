import os
import sys
from typing import Optional, Tuple

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
from cassis_utility.loading_utility import load_cas_from_xmi_dir, \
    load_cas_from_dir, \
    find_paths_in_dir, \
    load_typesystem_from_path, \
    load_cas_from_path

import cassis
from tqdm import tqdm
from datetime import datetime, timedelta


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def bucket_bundestag(input_tuple: Tuple[cassis.Cas, Optional[str]]) -> str:
    """
    Getting year-bucket-identifier for cas-object from Bundestag-Corpus.
    :param input_tuple:
    :return:
    """

    cas, _ = input_tuple
    # ==== Loading Type DocumentAnnotation ====
    document_annotations = cas.select("org.texttechnologylab.annotation.DocumentAnnotation")

    # ==== Getting Year ====
    print(document_annotations)
    timestamp = int(document_annotations[0]["timestamp"])
    dt = datetime(1970, 1, 1) + timedelta(milliseconds=timestamp)
    year = dt.year
    return str(year)


def bucket_coah(input_tuple: Tuple[cassis.Cas, Optional[str]]) -> str:
    """
    Getting year-bucket-identifier for cas-object from COAH-Corpus.
    :param input_tuple:
    :return:
    """
    cas, _ = input_tuple
    # ==== Loading Type DocumentMetaData ====
    document_metadata = cas.select("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")

    # ==== Getting Year ====
    title = document_metadata[0]["documentTitle"]
    return title.split("_")[1]


def bucket_dta(input_tuple: Tuple[cassis.Cas, Optional[str]]) -> str:
    """
    Getting year-bucket-identifier for cas-object from DTA-Corpus.
    :param input_tuple:
    :return:
    """
    cas, _ = input_tuple
    # ==== Loading Type DocumentMetaData ====
    document_metadata = cas.select("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")

    # ==== Getting Year ====
    id = document_metadata[0]["documentId"]
    year = id.split(".")[0].split("_")[-1]
    return year


def bucket_hansard(input_tuple: Tuple[cassis.Cas, Optional[str]]) -> str:
    """
    Getting year-bucket-identifier for cas-object from Hansard-Corpus.
    :param input_tuple:
    :return:
    """
    cas, _ = input_tuple
    # ==== Loading Type DocumentAnnotation ====
    document_annotations = cas.select("org.texttechnologylab.annotation.DocumentAnnotation")
    year = document_annotations[0]["dateYear"]
    return str(year)


def load_cas_from_corpus(dir_path: str, corpus_ident: str, verbose: bool) -> dict:
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
        print(year)

        # TODO: Add extracting of measures for a cas object

        # ==== pushing result into buckets they belong ====
        # (TODO: Edit out the saving of the cas itself because of memory, instead save results of measures)
        if year in buckets:
            buckets[year].append((cas, file_paths[i]))
        else:
            buckets[year] = [(cas, file_paths[i])]
        bar.update(1)

    return buckets



load_cas_from_corpus("/resources/corpora/COHA/texts_clean_xmi_ttlab/text_1810s_kso", "COAH", True)

#load_cas_from_corpus("/resources/corpora/paraliamentary_german/xmi_ttlab", "", True)
