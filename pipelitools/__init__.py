"""
Tools for data analysis and modeling.
"""

__version__ = "1.1.2"

from pipelitools import preprocessing, models

# from pipelitools.models import clustering
# from pipelitools.models import classification

from importlib import reload
reload(preprocessing)
reload(models)
# reload(outliers)
# reload(metrics)
# reload(clustering)
# reload(features)
# reload(classification)

# from pipelitools.preprocessing.eda import Dataset
# from pipelitools.preprocessing.features import *
# from pipelitools.models.classification import *
from .utils import check, log, black, green, red, yellow

# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# print('changed')