import errno
import pathlib
import shutil
import zipfile
from typing import List, Union

import numpy as np
from tqdm import tqdm
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, Command, Package, Subsubsection
from pylatex.utils import italic, NoEscape
import os
import pdflatex
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def zip_dir(dir: Union[pathlib.Path, str], filename: Union[pathlib.Path, str]):
    """Zip the provided directory without navigating to that directory using `pathlib` module"""

    # Convert to Path object
    dir = pathlib.Path(dir)

    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in dir.rglob("*"):
            zip_file.write(entry, entry.relative_to(dir))


def find_all_tex_files(path: str) -> [str]:
    """
    Function for finding all .tex files in a directory.
    :param path:
    :return:
    """
    dir_stack = []
    tex_files = []
    while True:
        names = [name for name in os.listdir(path)]
        for name in names:
            path_name = os.path.join(path, name)
            if os.path.isdir(path_name):
                dir_stack.append(path_name)
            else:
                if path_name[-4:] == ".tex":
                    tex_files.append(path_name)

        if not dir_stack:
            break
        else:
            path = dir_stack.pop()
    return tex_files


def copy_data_files_to_destination(filepaths: List[str],
                                   destination_path: str):
    """
    Function for creating new directory with all tex files copied.
    destination path has to contain "data".
    :param filepaths:
    :param destination_path:
    :return:
    """
    # --> creating destination path:
    try:
        # --> delete:
        shutil.rmtree(destination_path)
    except:
        pass
    # --> create new one:
    pathlib.Path(destination_path).mkdir(parents=True, exist_ok=True)

    new_filepaths = []
    for i in range(0, len(filepaths)):
        temp = filepaths[i].split("/data/")[-1].split("/", 1)
        temp[-1] = temp[-1].replace("/", "_")
        temp = "/".join(temp)
        new_filepaths.append(os.path.join(destination_path, temp))

    pbar = tqdm(desc="Saving single data-tex-files", total=len(filepaths))
    for i in range(0, len(filepaths)):
        src = filepaths[i]
        dest = new_filepaths[i]
        try:
            shutil.copy(src, dest)
        except IOError as e:
            # ENOENT(2): file does not exist, raised also on missing dest parent dir
            if e.errno != errno.ENOENT:
                raise
            # try creating parent directories
            os.makedirs(os.path.dirname(dest))
            shutil.copy(src, dest)
        pbar.update(1)


def save_whole_eger_paper_results(local_path: bool = False):

    # ==== Different Corpora ====
    corpora = ["Hansard", "COAH", "DTA", "Bundestag"]

    # ==== Making Fodler for results ====
    project_path = os.path.join(ROOT_DIR, "src/latex/eger_paper_results")
    # --> if exists delete to create new one:
    try:
        # --> delete:
        shutil.rmtree(project_path)
    except:
        pass
    # --> create new one:
    pathlib.Path(project_path).mkdir(parents=True, exist_ok=True)

    # --> Saving .tex for all corpora:
    for corpus in corpora:
        export_results_as_one_tex(corpus_ident=corpus, local_path=local_path)

    # --> Saving all .tex files:
    file_paths = find_all_tex_files(path=os.path.join(ROOT_DIR, "data"))
    copy_data_files_to_destination(filepaths=file_paths, destination_path=os.path.join(project_path, "data"))
    zip_dir(project_path, os.path.join(ROOT_DIR, "src/latex/eger_paper_results.zip"))

def export_results_as_one_tex(corpus_ident: str,
                              local_path: bool = False) -> None:
    """
    Function Saves Results of all calculations to one big latex project.
    :param corpus_ident:
    :param local_path:
    :return:
    """

    # ==== Latex Document ====
    doc = Document(documentclass="article",
                   lmodern=True)

    # ==== Adding Header/Preamble ====
    # --> add package:
    doc.packages.append(Package("pgfplots"))
    doc.packages.append(Package("graphicx"))
    doc.packages.append(Package("caption"))
    doc.packages.append(Package("sidecap"))
    doc.packages.append(Package("subcaption"))
    # --> add misc:
    doc.preamble.append(Command('DeclareUnicodeCharacter', arguments=["2212", "-"]))
    doc.preamble.append(Command('usepgfplotslibrary', arguments="groupplots,dateplot"))
    doc.preamble.append(Command('usetikzlibrary', arguments="patterns,shapes.arrows"))
    doc.preamble.append(Command('pgfplotsset', arguments="compat=newest"))
    # --> add preamble:
    doc.preamble.append(Command('title', 'Graphics Eger Paper'))
    doc.preamble.append(Command('author', 'Leon Hammerla'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    # --> maketitle command:
    doc.append(NoEscape(r'\maketitle'))
    doc.append(NoEscape(r"\newpage"))

    # ==== Adding Result-Plots ====
    # --> different measurements to display:
    relevant_measurements = ["n_verbs0", "complexity_ratio1", "absolute_complexity2", "order_of_tree3",
                             "dependency_index4", "stratum_of_tree5", "depth_of_tree6",
                             "ratio_vertices_on_lp_from_rt7", "leaf_distance_entropy8",
                             "ratio_of_leafs_at_d1_to_rt9", "mean_dependency_distance10",
                             "dependency_distance_entropy11", "ratio_arcs_adjacent_tokens12",
                             "ratio_arcs_distances_occ_once13", "imbalance_index14", "ratio_of_leafs15",
                             "number_of_leafs16", "width_of_tree19",
                             "lowest_lv_max_width20", "ratio_vertices_belonging_latter_level21",
                             "Hirsch_index22", "ratio_vertices_contributing_h_index23", "relative_h_index24"]
    # --> different sentence lengths that were observed:
    sent_lengths = ["sent_length_02-09", "sent_length_10-19", "sent_length_20-29", "sent_length_30-39", "sent_length_40+"]
    # --> creating pbar:
    pbar = tqdm(total=2*len(relevant_measurements), desc=f"Saving {corpus_ident}")
    # --> writing Sent&Doc-Based to tex:
    for res_type in ["sent", "doc"]:
        with doc.create(Section(f"{res_type}-Based:")):
            for measurement in relevant_measurements:
                with doc.create(Subsection(NoEscape(measurement.replace("_", " ")))):
                    # --> Creating Figure:
                    with doc.create(Figure(position='ht')) as fig:
                        # --> basic figure structure:
                        fig.append(NoEscape(r"\centering"))
                        fig.append(
                            Command("setlength", arguments=[NoEscape(r"\abovecaptionskip"), NoEscape(r"-35pt")]))
                        # --> adding single subplots to figure:
                        c = 0
                        for sent_length in sent_lengths:
                            # --> getting path:
                            if local_path:
                                measurement_plot_path = os.path.join(ROOT_DIR, f"data/{corpus_ident}/{res_type}/{sent_length}/tex_files/{measurement}.tex")
                            else:
                                measurement_plot_path = f"data/{corpus_ident}/{res_type}_{sent_length}_tex_files_{measurement}.tex"

                            # fig.append(Command("caption", arguments="Testoo"))
                            fig.append(Command("subfloat", options=NoEscape(sent_length.replace("_", " ")),
                                                arguments=Command("scalebox", arguments=["0.6", Command("input",
                                                                                                        arguments=NoEscape(measurement_plot_path))])))
                            c += 1
                            # --> Grouping figs two in one row:
                            if c % 2 == 0:
                                fig.append(NoEscape(r"\\"))

                    doc.append(NoEscape(r"\newpage"))

                    # --> creating figure for acf-statistics:
                    with doc.create(Subsubsection("Statistics-acf")):
                        with doc.create(Figure(position='ht')) as fig:
                            # --> basic figure structure:
                            fig.append(NoEscape(r"\centering"))
                            fig.append(
                                Command("setlength", arguments=[NoEscape(r"\abovecaptionskip"), NoEscape(r"-35pt")]))
                            # --> adding single subplots to figure:
                            c = 0
                            for sent_length in sent_lengths:
                                # --> getting path:
                                if local_path:
                                    acf_plot_path = os.path.join(ROOT_DIR, f"data/{corpus_ident}/{res_type}/{sent_length}/statistics/{measurement}/acf.tex")
                                else:
                                    acf_plot_path = f"data/{corpus_ident}/{res_type}_{sent_length}_statistics_{measurement}_acf.tex"


                                # fig.append(Command("caption", arguments="Testoo"))
                                fig.append(Command("subfloat", options=NoEscape(sent_length.replace("_", " ")),
                                                   arguments=Command("scalebox", arguments=["0.6", Command("input",
                                                                                                           arguments=NoEscape(
                                                                                                               acf_plot_path))])))
                                c += 1
                                # --> Grouping figs two in one row:
                                if c % 2 == 0:
                                    fig.append(NoEscape(r"\\"))

                    doc.append(NoEscape(r"\newpage"))

                    # --> creating figure for pacf-statistics:
                    with doc.create(Subsubsection("Statistics-pacf")):
                        with doc.create(Figure(position='ht')) as fig:
                            # --> basic figure structure:
                            fig.append(NoEscape(r"\centering"))
                            fig.append(
                                Command("setlength", arguments=[NoEscape(r"\abovecaptionskip"), NoEscape(r"-35pt")]))
                            # --> adding single subplots to figure:
                            c = 0
                            for sent_length in sent_lengths:
                                # --> getting path:
                                if local_path:
                                    pacf_plot_path = os.path.join(ROOT_DIR, f"data/{corpus_ident}/{res_type}/{sent_length}/statistics/{measurement}/pacf.tex")
                                else:
                                    pacf_plot_path = f"data/{corpus_ident}/{res_type}_{sent_length}_statistics_{measurement}_pacf.tex"

                                # fig.append(Command("caption", arguments="Testoo"))
                                fig.append(Command("subfloat", options=NoEscape(sent_length.replace("_", " ")),
                                                   arguments=Command("scalebox", arguments=["0.6", Command("input",
                                                                                                           arguments=NoEscape(
                                                                                                               pacf_plot_path))])))
                                c += 1
                                # --> Grouping figs two in one row:
                                if c % 2 == 0:
                                    fig.append(NoEscape(r"\\"))

                    doc.append(NoEscape(r"\newpage"))

                    # --> writing correlation to .tex file:
                    with doc.create(Subsubsection("Statistics-cor")):
                        doc.append(NoEscape(r"\noindent"))
                        for sent_length in sent_lengths:
                            cor_path = os.path.join(ROOT_DIR, f"data/{corpus_ident}/{res_type}/{sent_length}/statistics/{measurement}/single_value_measurements.txt")
                            with open(cor_path, "r") as f:
                                cor = f.readlines()[0].split("=")[-1]
                                doc.append(NoEscape(rf"{sent_length.replace('_', ' ')} - Correlation: {cor}\\"))

                    doc.append(NoEscape(r"\newpage"))

                pbar.update(1)
                pbar.refresh()

    doc.generate_pdf(os.path.join(ROOT_DIR, f"src/latex/eger_paper_results/{corpus_ident}"), clean_tex=False, compiler='pdflatex')
    # ==== Export .tex file ====
    # doc.generate_tex(os.path.join(ROOT_DIR, f"src/latex/eger_paper_results/{corpus_ident}"))




if __name__ == '__main__':
    """
    image_filename1 = os.path.join(ROOT_DIR, "data/Hansard/sent/sent_length_20-29/tex_files/mean_dependency_distance10.tex")
    image_filename2 = os.path.join(ROOT_DIR,
                                  "data/Hansard/sent/sent_length_20-29/tex_files/complexity_ratio1.tex")
    image_filename3 = os.path.join(ROOT_DIR,
                                  "data/Hansard/sent/sent_length_20-29/tex_files/dependency_index4.tex")
    image_filename4 = os.path.join(ROOT_DIR,
                                  "data/Hansard/sent/sent_length_20-29/tex_files/absolute_complexity2.tex")

    # geometry_options = {"tmargin": "1cm", "lmargin": "10cm"}
    # ==== Latex Document ====
    doc = Document(documentclass="article",
                   lmodern=True)

    # ==== Adding Header/Preamble ====
    # --> add package:
    doc.packages.append(Package("pgfplots"))
    doc.packages.append(Package("graphicx"))
    doc.packages.append(Package("caption"))
    doc.packages.append(Package("sidecap"))
    doc.packages.append(Package("subcaption"))
    # --> add misc:
    doc.preamble.append(Command('DeclareUnicodeCharacter', arguments=["2212", "-"]))
    doc.preamble.append(Command('usepgfplotslibrary', arguments="groupplots,dateplot"))
    doc.preamble.append(Command('usetikzlibrary', arguments="patterns,shapes.arrows"))
    doc.preamble.append(Command('pgfplotsset', arguments="compat=newest"))
    # --> add preamble:
    doc.preamble.append(Command('title', 'Graphics Eger Paper'))
    doc.preamble.append(Command('author', 'Leon Hammerla'))
    doc.preamble.append(Command('date', NoEscape(r'\today')))
    # --> maketitle command:
    doc.append(NoEscape(r'\maketitle'))

  

    with doc.create(Section("test")):
        with doc.create(Figure(position='ht')) as fig:
            fig.append(NoEscape(r"\centering"))
            fig.append(Command("setlength", arguments=[NoEscape(r"\abovecaptionskip"), NoEscape(r"-35pt")]))
            # fig.append(Command("caption", arguments="Testoo"))
            fig.append(Command("subfloat", options=NoEscape(r"sub - caption"),
                               arguments=Command("scalebox", arguments=["0.6", Command("input", arguments=NoEscape(image_filename1))])))

            fig.append(Command("subfloat", options=NoEscape(r"sub - caption"),
                               arguments=Command("scalebox", arguments=["0.6", Command("input", arguments=NoEscape(
                                   image_filename1))])))
            fig.append(NoEscape(r"\\"))
            fig.append(Command("subfloat", options=NoEscape(r"sub - caption"),
                               arguments=Command("scalebox", arguments=["0.6", Command("input", arguments=NoEscape(
                                   image_filename1))])))
            fig.append(Command("subfloat", options=NoEscape(r"sub - caption"),
                               arguments=Command("scalebox", arguments=["0.6", Command("input", arguments=NoEscape(
                                   image_filename1))])))
            fig.append(NoEscape(r"\\"))
            fig.append(Command("subfloat", options=NoEscape(r"sub - caption"),
                               arguments=Command("scalebox", arguments=["0.6", Command("input", arguments=NoEscape(
                                   image_filename1))])))

    doc.generate_pdf(os.path.join(ROOT_DIR, "src/latex/test01"), clean_tex=False, compiler='pdflatex')
    """
    #export_results_as_one_tex("Hansard")

    save_whole_eger_paper_results(local_path=True)