import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("./aclImdb/imdb_te.csv", encoding='charmap')
print(df.head())