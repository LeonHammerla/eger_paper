import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from tqdm import tqdm
import random
from treelib import Tree, Node
import math
from collections import Counter
import tikzplotlib


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(ROOT_DIR, "data", "test")




"""#dir_path = os.path.join(data_dir, "COAH", "timeslices")
#filepaths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]
filepaths = ["hooboo/hbjkbuhk/ozunbou/res_929.pickle",
             "hooboo/hbjkbuhk/ozunbou/path_72.pickle",
             "hooboo/hbjkbuhk/ozunbou/path_929.pickle",
             "hooboo/hbjkbuhk/ozunbou/path_1.pickle",
             "hooboo/hbjkbuhk/ozunbou/res_1.pickle",
             "hooboo/hbjkbuhk/ozunbou/res_72.pickle",
             "hooboo/hbjkbuhk/ozunbou/path_2.pickle",
             "hooboo/hbjkbuhk/ozunbou/res_2.pickle",
             ]

filepaths.sort(key=lambda x: int(x.split("/")[-1].split("_")[-1].rstrip(".pickle")) if "res_" in x else int(x.split("/")[-1].split("_")[-1].rstrip(".pickle")) + 0.5)

print(filepaths)


for i in range(0, 10, 2):
    print(i)

print(type((2,2)))"""

with PdfPages(os.path.join(data_dir, 'foo.pdf')) as pdf:
    #plt.semilogx()
    flierprops = dict(marker='x', markersize=2,
                      linestyle='none')
    a = {}
    for i in range(1900, 1960):
        x = []
        for j in range(0, 10):
            x.append(random.randrange(0, 100))
        a[str(i)] = x
    #a["1939"][4] = 10000
    df = pd.DataFrame(a)

    fig = df.plot(xlabel='Time_Slices', ylabel='Token_per_sent', kind="box", fontsize=4, flierprops=flierprops, showfliers=True, rot=90).get_figure()

    #plt.xticks(rotation=90)

    #fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box")
    #[ax_tmp.set_xlabel('') for ax_tmp in np.asarray(fig).reshape(-1)]
    #fig = np.asarray(fig).reshape(-1)[0].get_figure()
    pdf.savefig(fig)
    tikzplotlib.save(filepath=os.path.join(data_dir, "0.tex"), figure=fig, extra_axis_parameters=["font={\\fontsize{3}{12}\selectfont}"])


    print(type(fig))
    df = pd.DataFrame({"h": [1.2, 1.2121212, 3,5,np.nan], "x": [1, 2, 3, 8, 9]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    #plt.xticks(rotation=90)
    pdf.savefig(fig)
    tikzplotlib.save(filepath=os.path.join(data_dir, "1.tex"), figure=fig, axis_height="8cm", axis_width="16cm")
