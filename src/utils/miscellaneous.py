import re
import networkx as nx
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def from_camel_case_to_underscore(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def color_map(vmin, vmax, reverse=False):

    if reverse:
        cmap = cm.coolwarm_r
    else:
        cmap = cm.coolwarm
    norm = Normalize(vmin, vmax)

    def cmap_(x):
        return cmap(norm(x))
    return cmap_