import os
import sys
from typing import Optional, Tuple

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))

import cassis
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

    # ==== Getting Year ====
    year = document_annotations[0]["dateYear"]
    return str(year)
