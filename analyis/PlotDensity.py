import argparse
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument("filename")
cmdLineArgs = cmdLineParser.parse_args()

hf = h5py.File(cmdLineArgs.filename, "r")
n1 = hf.get("/").get("samples")
n2 = np.array(n1)

x = n2[0, :]
y = n2[1, :]

# adapted from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
nullfmt = NullFormatter()

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.05

axScatter.set_xlim((0, 1))
axScatter.set_ylim((0, 1))

bins = np.arange(0, 1 + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation="horizontal")

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()
