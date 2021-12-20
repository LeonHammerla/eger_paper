import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(ROOT_DIR, "data")


with PdfPages(os.path.join(data_dir, 'foo.pdf')) as pdf:
    plt.semilogy()
    df = pd.DataFrame({"h":[1.2, 1.2121212, 3], "x":[1,2,3]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    pdf.savefig(fig)

    print(type(fig))
    df = pd.DataFrame({"h": [1.2, 1.2121212, 3,5,np.nan], "x": [1, 2, 3, 8, 9]})
    fig = df.plot(xlabel ='Time_Slices', ylabel='Token_per_sent', kind="box").get_figure()
    pdf.savefig(fig)








#plt.semilogy()
#print(type(axxr))
#plt.show()


"""data = np.array([[1.2,1], [1.2121212, 2], [3, 3], [np.nan, 5], [np.nan, 6]])
print(data)
#data[30, 0] = np.NaN
#data[20, 1] = np.NaN

# Filter data using np.isnan
mask = ~np.isnan(data)
data = [d[m] for d, m in zip(data.T, mask.T)]


# basic plot
plt.boxplot(data)

plt.show()"""