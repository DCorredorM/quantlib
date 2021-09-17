import re
import networkx as nx
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from datetime import date
import pandas as pd
import datetime


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


def string_date_to_datetime(string_date):
    """

    Parameters
    ----------
    string_date: month day year

    Returns
    -------

    """
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

    month, day, year = string_date.split(' ')
    month = months.index(month.capitalize()) + 1

    day = int(day)
    year = int(year)
    return date(year, month, day)


def to_date(date_):
    if isinstance(date_, pd._libs.tslibs.timestamps.Timestamp) or isinstance(date_, datetime.datetime):
        return date_.date()
    elif isinstance(date_, date):
        return date_








