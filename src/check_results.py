#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(file_path):
    pass

def plot_results(loss):
    fig, ax = plt.subplots(fig_size=(12, 8))
    sns.lineplot(pd.DataFrame(loss), ax=ax)
