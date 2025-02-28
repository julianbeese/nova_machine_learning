import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from scipy.stats import mstats
from xgboost import XGBRegressor

df_train = pd.read_csv("C:/Users/maxpi/Desktop/Universität/Machine Learning/groupproject/archive (2)/train.csv") # insert file path
df_test = pd.read_csv("C:/Users/maxpi/Desktop/Universität/Machine Learning/groupproject/archive (2)/test.csv") # insert file path

df = pd.concat([df_test,df_train], ignore_index=True)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')