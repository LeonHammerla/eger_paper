import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
from tqdm import tqdm


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(ROOT_DIR, "data")

"""
dir_path = os.path.join(data_dir, "COAH", "timeslices")
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

print(type((2,2)))

with PdfPages(os.path.join(data_dir, 'foo.pdf')) as pdf:
    plt.semilogy()
    df = pd.DataFrame({"h":[1.2, 1.2121212, 3], "x":[1,2,3]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    pdf.savefig(fig)

    print(type(fig))
    df = pd.DataFrame({"h": [1.2, 1.2121212, 3,5,np.nan], "x": [1, 2, 3, 8, 9]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    pdf.savefig(fig)


plt.semilogy()
print(type(axxr))
plt.show()
data = np.array([[1.2,1], [1.2121212, 2], [3, 3], [np.nan, 5], [np.nan, 6]])
print(data)
#data[30, 0] = np.NaN
#data[20, 1] = np.NaN

# Filter data using np.isnan
mask = ~np.isnan(data)
data = [d[m] for d, m in zip(data.T, mask.T)]


# basic plot
plt.boxplot(data)

plt.show()
"""
import time
def f(i):
    time.sleep(i)
pool = Pool(5)
a = [2 for i in range(0, 10)]

with pool as p:
    pbar = tqdm(total=len(a))
    for res in p.imap_unordered(f, a):
        pbar.update(1)

pool.close()
pool.join()