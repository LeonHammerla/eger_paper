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


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(ROOT_DIR, "data", "test")


tree = Tree()
tree.create_node("Harry", "harry")  # root node
tree.create_node("Jane", "jane", parent="harry")
tree.create_node("Bill", "bill", parent="harry")
tree.create_node("Diane", "diane", parent="jane")
tree.create_node("Mary", "mary", parent="diane")
tree.create_node("Mark", "mark", parent="jane")
tree.create_node("Leon", 0, parent="bill")


tree.show()
"""print(type(tree.leaves()[0]))
print(type(tree))
print(tree.parent("jane"))
print(tree.is_branch("mark"))
#print(tree.link_past_node("jane"))
print(tree.paths_to_leaves())
print(tree.ancestor("diane"))
print(tree.level("harry"))
print(tree.level("mark"))
print(tree.is_ancestor("bill", "mark"))
print(tree.is_ancestor("bill", 0))
print(tree.to_dict())
print(tree.all_nodes())

"""

def dist(node1, node2, tree):
    name1 = node1.identifier
    name2 = node2.identifier
    if tree.is_ancestor(name1, name2):
        return tree.level(name2) - tree.level(name1)
    elif tree.is_ancestor(name2, name1):
        # return tree.level(name1) - tree.level(name2)
        return math.inf
    else:
        return math.inf

"""print(dist(tree.get_node("bill"), tree.get_node("mark"), tree))

print(dist(tree.get_node("harry"), tree.get_node("mark"), tree))
print(dist(tree.get_node("harry"), tree.get_node("mark"), tree))

print(dist(tree.get_node("bill"), tree.get_node(0), tree))
print(dist(tree.get_node("mark"), tree.get_node(0), tree))

print(dist(tree.get_node("jane"), tree.get_node("mark"), tree))
print(dist(tree.get_node("mark"), tree.get_node("jane"), tree))

"""
dist_dict = {}
all_nodes = tree.all_nodes()
for i in all_nodes:
    a = []
    for j in all_nodes:
        a.append((dist(i, j, tree), j))
    b = dict()
    for j in a:
        if j[0] in b:
            b[j[0]].append(j[-1])
        else:
            b[j[0]] = [j[-1]]
    dist_dict[i] = b

for item in dist_dict.items():
    print(item[0])
    print(item[1])
    print("===================================================================")

def degree(node, dist_dict):
    print(node)
    try:
        return len(dist_dict[node][1])
    except:
        return 0
print("==============")
print(tree.depth())
print("==============")

for i in all_nodes:
    print(degree(i, dist_dict))
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

    print(type(fig))
    df = pd.DataFrame({"h": [1.2, 1.2121212, 3,5,np.nan], "x": [1, 2, 3, 8, 9]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    #plt.xticks(rotation=90)
    pdf.savefig(fig)

"""

"""
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
"""