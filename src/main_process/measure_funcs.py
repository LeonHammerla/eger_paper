import os
import sys
from collections import Counter
from copy import copy
from typing import Optional, Tuple, List, Dict

from tqdm import tqdm

sys.path.append("/home/stud_homes/s5935481/uima_cassis/src")
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
import cassis
from operator import indexOf
from cassis_utility.loading_utility import load_typesystem_from_path, load_cas_from_path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
VERB_TAGS = ["VAPP", "VAPS", "VMFIN", "VMIMP", "VMINF", "VMPP", "VMPS", "VVFIN", "VVIMP", "VVINF", "VVPP", "VVPS"]


def select_sentences_from_cas(cas: cassis.Cas) -> List[cassis.typesystem.FeatureStructure]:
    """
    Function returns all sentences from a cas-object.
    :param cas:
    :return:
    """
    sentences = cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
    return sentences


def select_dependencies_from_sentence(cas: cassis.Cas,
                                      sentence: cassis.typesystem.FeatureStructure) -> List[cassis.typesystem.FeatureStructure]:
    """
    Function returns list of dependencies in a given sentence from a cas.
    :param cas:
    :param sentence:
    :return:
    """
    dependencies = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", sentence)
    return dependencies

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


def n_verbs_in_sentence(cas: cassis.Cas,
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
        is_verb = False
        pos = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", token)[0]
        pos_value = pos["PosValue"]
        for possible_verb_tag in VERB_TAGS:
            if possible_verb_tag in pos_value:
                is_verb = True
                break
        if not is_verb:
            pos_value2 = pos["coarseValue"]
            if pos_value2:
                if pos_value2.lower() == "verb":
                    is_verb = True
        if is_verb:
            verb_count += 1

    return verb_count


def n_token_in_sentence(tokens: List[cassis.typesystem.FeatureStructure]) -> int:
    """
    Calculates how many tokens are in a sentence.
    :param tokens:
    :return:
    """
    return len(tokens)


def mdd_of_sent(tokens: List[cassis.typesystem.FeatureStructure],
                dependencies: List[cassis.typesystem.FeatureStructure]) -> float:
    """
    For a Pair of dependent and governor the distance of their position
    in a sentence is calculated.
    The mdd ist the mean for all those pairs in a sentence.

    Definition:
    MDD(sent) = 1 / (n - 1) * sum(DD:0...DD:n-1)

    Here n is the number of words in the sentence and DDi is the dependency distance of the
    i-th syntactic link of the sentence. Usually in a sentence there is one word (the root verb)
    without a head, whose DD is defined as zero.
    :param dependencies:
    :param tokens:
    :return:
    """
    tokens = [token["begin"] for token in tokens]
    position_sum = 0
    for dependency in dependencies:
        if (dependency["DependencyType"] == "--") or ("root" in dependency["DependencyType"].lower()):
            pass
        else:
            dependent = dependency["Dependent"]
            governor = dependency["Governor"]
            position_sum += abs(indexOf(tokens, dependent["begin"]) - indexOf(tokens, governor["begin"]))

    assert len(tokens) == len(dependencies), "Not every Token has a Dependency-Annotation!!!"
    #print(position_sum)
    return position_sum / (len(tokens) - 1) if (len(tokens) - 1) > 0 else 0.0


def altmann_of_sent(tokens: List[cassis.typesystem.FeatureStructure]) -> float:
    """
    MAL states that with increasing size of a linguistic construct, the size of its parts shrinks.
    ??? Calculation is the mean of the size of its parts ???
    :param tokens:
    :return:
    """
    sum_length_of_subconstructs = 0
    for token in tokens:
        sum_length_of_subconstructs += len(token.get_covered_text())
    return sum_length_of_subconstructs / len(tokens)


def calc_dep_depth_of_dependency(cas: cassis.Cas,
                                 dependency: cassis.typesystem.FeatureStructure) -> int:
    """
    Function Calculates depth of given dependency in the dependency-tree.
    :param cas:
    :param dependency:
    :return:
    """
    gov = dependency
    depth = 0
    while True:
        next_gov = cas.select_covered("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", gov["Governor"])[0]
        if gov["begin"] == next_gov["begin"]:
            break
        else:
            gov = copy(next_gov)
            depth += 1
    return depth


def calc_max_dep_depth_of_sentence_and_depthmap(cas: cassis.Cas,
                                                dependencies: List[cassis.typesystem.FeatureStructure]) -> Tuple[int, Dict[int, int]]:
    """
    Function calculates max depth of dependency tree of a sentence and returnes a dict with a count of every d
    :param cas:
    :param dependencies:
    :return:
    """
    max_depth = 0
    depth_list = []
    for dependency in dependencies:
        depth = calc_dep_depth_of_dependency(cas, dependency)
        if depth > max_depth:
            max_depth = depth
        depth_list.append(depth)
    return max_depth, dict(Counter(depth_list))


def sent_based_measurements_for_cas(cas: cassis.Cas,
                                    verbose: bool = False) -> List[Tuple[int, int, int, float]]:
    """
    Function for calculating all measurements for a given cas-object, sentence
    by sentence and returning result-tuple for each sentence in a list.
    :param verbose:
    :param cas:
    :return:
    """

    results = []

    # ==== Going for each sentence in cas ====
    sentences = select_sentences_from_cas(cas)

    if verbose:
        pbar = tqdm(total=len(sentences), desc="Calculating_sent_based", leave=True, disable=False, position=0)
    else:
        pbar = tqdm(total=len(sentences), desc="Calculating_sent_based", leave=True, disable=True, position=0)

    for sentence in sentences:
        tokens = select_tokens_of_sentence_from_cas(cas, sentence)
        dependencies = select_dependencies_from_sentence(cas, sentence)

        # ==== Calculating measurements ====
        n_toks = n_token_in_sentence(tokens)
        n_verbs = n_verbs_in_sentence(cas, tokens)
        mdd = mdd_of_sent(tokens, dependencies)
        max_depth, _ = calc_max_dep_depth_of_sentence_and_depthmap(cas, dependencies)

        # TODO: Add Altmann-Measure
        # altmann = altmann_of_sent(tokens)

        # ==== Adding result tuple for each sentence to total result ====
        results.append((n_toks, n_verbs, max_depth, mdd))

        pbar.update(1)

    return results


def doc_based_measurements_for_cas(results: List[Tuple[int, int, int, float]]) -> Tuple[int, int, int, float, float, float, float]:
    """
    Function for converting sentence based list of result tuples of one cas to one doc based
    result tuple.
    :param results:
    :return:
    """
    # TODO: Adding Altmann measurement
    n_sents = len(results)
    n_token = 0
    n_verbs = 0
    sum_max_depth = 0
    weighted_mdd_sum = 0
    for i in range(0, n_sents):
        n_token += results[i][0]
        n_verbs += results[i][1]
        sum_max_depth += results[i][2]

        # ==== Calculate MDD as weighted Average for a Doc as in: ====
        # (The effects of sentence length on dependency distance, dependency direction and the implications–Based on a parallel English–Chinese dependency treebank)
        weighted_mdd_sum += (results[i][3] * (results[i][0] - 1))


    mdd = weighted_mdd_sum / (n_token - n_sents)
    tok_per_sentence = n_token / n_sents
    v_per_sentence = n_verbs / n_sents
    avg_max_depth = sum_max_depth / n_sents

    return n_token, n_verbs, n_sents, tok_per_sentence, v_per_sentence, mdd, avg_max_depth


if __name__ == '__main__':
    typesystem_path = os.path.join(ROOT_DIR, "TypeSystem.xml")
    typesystem = load_typesystem_from_path(typesystem_path)
    c = load_cas_from_path(filepath="/vol/s5935481/eger_resources/DTA/dta_kernkorpus_2020-07-20/ttlab_xmi/humboldt_kosmos03_1850.TEI-P5.xml/humboldt_kosmos03_1850.TEI-P5.xml#1.xmi.gz", typesystem=typesystem)
    #c = load_cas_from_path(filepath="/resources/corpora/paraliamentary_german/xmi_ttlab/LL2/10_1/272.txt.xmi.gz", typesystem=typesystem)
    res = sent_based_measurements_for_cas(c, verbose=True)
    for i in res:
        print(i)
    res2 = doc_based_measurements_for_cas(res)
    print(res2)