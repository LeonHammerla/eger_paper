import os
import sys
from typing import Optional, Tuple, List
sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cassis
from operator import indexOf

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
VERB_TAGS = ["VAPP", "VAPS", "VMFIN", "VMIMP", "VMINF", "VMPP", "VMPS", "VVFIN", "VVIMP", "VVINF", "VVPP", "VVPS"]


def select_sentences_from_cas(cas:cassis.Cas) -> List[cassis.typesystem.FeatureStructure]:
    """
    Function returns all sentences from a cas-object.
    :param cas:
    :return:
    """
    sentences = cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    return sentences


def select_tokens_of_sentence_from_cas(cas: cassis.Cas,
                                       sentence: cassis.typesystem.FeatureStructure) -> List[cassis.typesystem.FeatureStructure]:
    """
    Funmction returns all token from a sentence from a cas-object
    :param cas:
    :param sentence:
    :return:
    """
    tokens = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", sentence)
    return tokens


def verbs_per_sentence(cas: cassis.Cas,
                       tokens: List[cassis.typesystem.FeatureStructure]) -> int:
    """
    Calculates how many verbs are in a sentence.
    =========================== List of Verb-Tags ============================
    VAPP	Auxiliar, Partizip Präteritum, im Verbalkomplex
    VAPS	Auxiliar, Partizip Präsens, im Verbalkomplex
    VMFIN	Modalverb, finit
    VMIMP	Modalverb, Imperativ
    VMINF	Modalverb, Infinitiv
    VMPP	Modalverb, Partizip Präteritum, im Verbalkomplex
    VMPS	Modalverb, Partizip Präsens, im Verbalkomplex
    VVFIN	Vollverb, finit
    VVIMP	Vollverb, Imperativ
    VVINF	Vollverb, Infinitiv
    VVPP	Partizip Präteritum, im Verbalkomplex
    VVPS	Partizip Präsens, im Verbalkomplex
    ===========================================================================
    :param tokens:
    :param cas:
    :return:
    """

    verb_count = 0
    for token in tokens:
        pos = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", token)[0]
        pos_value = pos["PosValue"]
        for possible_verb_tag in VERB_TAGS:
            if possible_verb_tag in pos_value:
                verb_count += 1
                break

    return verb_count


def token_per_sentence(tokens: List[cassis.typesystem.FeatureStructure]) -> int:
    """
    Calculates how many tokens are in a sentence.
    :param tokens:
    :return:
    """
    return len(tokens)


def mdd_of_sent(cas: cassis.Cas,
                sentence: cassis.typesystem.FeatureStructure,
                tokens: List[cassis.typesystem.FeatureStructure]) -> float:
    """
    For a Pair of dependent and governor the distance of their position
    in a sentence is calculated.
    The mdd ist the mean for all those pairs in a sentence.

    Definition:
    MDD(sent) = 1 / (n - 1) * sum(DD:0...DD:n-1)

    Here n is the number of words in the sentence and DDi is the dependency distance of the
    i-th syntactic link of the sentence. Usually in a sentence there is one word (the root verb)
    without a head, whose DD is defined as zero.
    :param tokens:
    :param sentence:
    :param cas:
    :return:
    """
    position_sum = 0
    dependencies = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", sentence)
    for dependency in dependencies:
        if dependency["DependencyType"] == "--" or "root" in dependency["DependencyType"].lower():
            pass
        else:
            dependent = dependency["Dependent"]
            governor = dependency["Governor"]
            position_sum += abs(indexOf(tokens, dependent) - indexOf(tokens, governor))

    assert len(tokens) == len(dependencies), "Not every Token has a Dependency-Annotation!!!"

    return position_sum / (len(tokens) - 1)


def altmann_of_sent(tokens: List[cassis.typesystem.FeatureStructure]) -> float:
    """
    MAL states that with increasing size of a linguistic construct, the size of its parts shrinks.
    ??? Calculation is the mean of the size of its parts ???
    :param cas:
    :return:
    """
    sum_length_of_subconstructs = 0
    for token in tokens:
        sum_length_of_subconstructs += len(token.get_covered_text())
    return sum_length_of_subconstructs / len(tokens)


def measurements_for_cas_sent_based(cas: cassis.Cas) -> List[Tuple[int, int, float, float]]:
    """
    Function for calculating all measurements for a given cas-object, sentence
    by sentence and returning result-tuple for each sentence in a list.
    :param cas:
    :return:
    """

    results = []

    # ==== Going for each sentence in cas ====
    sentences = select_sentences_from_cas(cas)
    for sentence in sentences:
        tokens = select_tokens_of_sentence_from_cas(cas, sentence)

        # ==== Calculating measurements ====
        tok_per_sent = token_per_sentence(tokens)
        v_per_sent = verbs_per_sentence(cas, tokens)
        mdd = mdd_of_sent(cas, sentence, tokens)
        altmann = altmann_of_sent(tokens)

        # ==== Adding result tuple for each sentence to total result ====
        results.append((tok_per_sent, v_per_sent, mdd, altmann))

    return results


def measurements_for_cas_doc_based(results: List[Tuple[int, int, float, float]]):
    # TODO: Implement these, but have to clarify which should be used.
    n_sents = len(results)
    tok_per_sent = token_per_sentence(tokens)
    v_per_sent = verbs_per_sentence(cas, tokens)
    mdd = mdd_of_sent(cas, sentence, tokens)
    altmann = altmann_of_sent(tokens)
    for i in range(0, n_sents):
        results[i]