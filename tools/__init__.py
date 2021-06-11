"""
Tools for data analysis and modeling.
"""

__version__ = "1.0.0"

from datetime import datetime

from tools import preprocessing, models

# from tools.models import clustering
# from tools.models import classification

from importlib import reload
reload(preprocessing)
reload(models)
# reload(outliers)
# reload(metrics)
# reload(clustering)
# reload(features)
# reload(classification)

# from tools.preprocessing.eda import Dataset
# from tools.preprocessing.features import *
# from tools.models.classification import *
from .utils import check, log, black, green, red, yellow

# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# print('changed')