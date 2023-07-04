import numpy as np
import pandas as pd
from sklearn import metrics

# BirdCLEF-2023 used this metric with pad_rows=5.
# See https://www.kaggle.com/competitions/birdclef-2023/overview/evaluation.
# Smaller padding factors make sense for more balanced data, but using pad_rows=0
# causes problems if any label columns have no ones (i.e. a class with no occurrences).
def padded_cmap(solution, submission, pad_rows=1):
    ones = np.ones((1, solution.shape[1]))
    for i in range(pad_rows):
        solution = np.append(solution, ones, axis=0)
        submission = np.append(submission, ones, axis=0)

    return metrics.average_precision_score(solution, submission, average='macro')
