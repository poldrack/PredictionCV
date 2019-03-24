"""
Demonstration of common cross-validation bad practices
=======================================================

We show with a simple example pitfalls of a common practice:

- Using correlation as a score
- Computing scores on the predictions across scores

We generate a situation we there is no link between X and y, but were y
is made of the same number of -1, 1. In such a situation, LOO creates a
distribution of y_train for which the mean is anti-correlated with value
in the test set.

"""

import numpy as np
from sklearn import model_selection, linear_model, metrics

import pandas as pd
import seaborn
from matplotlib import pyplot as plt

n_samples = 50
n_boostraps = 200

rng = np.random.RandomState(seed=0)

# Generate data
def gen_data(n_samples):
    X = rng.randn(n_samples, 1)
    y = np.ones(n_samples)
    y[::2] = -1
    return X, y


model = linear_model.RidgeCV(fit_intercept=True)

scores = list()
cv_strategy = list()
measure = list()

for i in range(n_boostraps):
    X, y = gen_data(n_samples)
    X_val, y_val = gen_data(100000)

    scores.append(np.mean(
        model_selection.cross_val_score(model, X, y,
                                        cv=model_selection.KFold(5),
                                        scoring='r2')))
    cv_strategy.append('5-fold+ \nr2 in each fold')
    measure.append('cross-validation')

    scores.append(metrics.r2_score(y_val,
                                   model.fit(X, y).predict(X_val)))
    cv_strategy.append('5-fold+ \nr2 in each fold')
    measure.append('new data')

    scores.append(np.corrcoef(y,
        model_selection.cross_val_predict(model, X, y,
                                            cv=5))[0, 1])
    cv_strategy.append('LOO + \ncorrelation across folds')
    measure.append('cross-validation')

    scores.append(np.corrcoef(y_val,
                                   model.fit(X, y).predict(X_val))[0, 1])
    cv_strategy.append('LOO + \ncorrelation across folds')
    measure.append('new data')

    print('Done % 2i out of %i' % (i, n_boostraps))

scores = pd.DataFrame(dict(scores=scores, cv_strategy=cv_strategy,
                           measure=measure))

plt.figure(figsize=(5, 2))
seaborn.violinplot(x='scores', y='cv_strategy', data=scores,
                hue='measure', scale='width', orient='h', split=True)
plt.ylabel('')
plt.xlabel('')
plt.axvline(0, color='.1', zorder=0)
plt.tight_layout(pad=.01)
plt.legend(loc=(.51, .46), frameon=False)

