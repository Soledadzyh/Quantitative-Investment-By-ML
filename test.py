import csv
import datetime
from hashlib import new

import pandas as pd
import numpy as np
import time
data=csv.reader(open("test.csv",'r'))
# datay = pd.read_csv("texty.csv")
for i in data:
    print(i)