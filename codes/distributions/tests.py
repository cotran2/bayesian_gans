import numpy as np
import pandas as pd
import scipy
from scipy import stats
import math

from scipy import stats
class your_distribution(stats.rv_continuous):
    def _pdf(self, x):
        return np.exp(-x**2/2.)/np.sqrt(2.0*np.pi)

if __name__ == "__main__":
    distribution = your_distribution(name = "dis_1")
    print(distribution.rvs(size = 100))
