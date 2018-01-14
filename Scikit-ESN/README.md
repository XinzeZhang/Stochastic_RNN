# Scikit_ESN

[![Coverage Status](https://coveralls.io/repos/sylvchev/simple_esn/badge.svg?branch=master&service=github)](https://coveralls.io/github/sylvchev/simple_esn?branch=master)
[![Travis CI](https://travis-ci.org/sylvchev/simple_esn.svg?branch=master)](https://travis-ci.org/sylvchev/simple_esn)
[![Code Climate](https://codeclimate.com/github/sylvchev/simple_esn/badges/gpa.svg)](https://codeclimate.com/github/sylvchev/simple_esn)

## Scikit_ESN

**Scikit_ESN** implement a Python class of simple Echo State Networks models
witin the Scikit-learn framework. It is intended to be a fast-and-easy
transformation of an input signal in a reservoir of neurons. The classification could be done with any scikit-learn classifier/regressor.

The `Scikit_ESN` object could be part of a `Pipeline` and its parameter space could
be explored with a `GridSearchCV` for example.

The code is inspired by the "minimalistic ESN example" proposed by Mantas
Lukoševičius. It is licenced under GPLv3.

## Useful links

-   Code from Mantas Lukoševičius: http://organic.elis.ugent.be/software/minimal
-   Code from Mantas Lukoševičius: http://minds.jacobs-university.de/mantas/code
-   More serious reservoir computing softwares: http://organic.elis.ugent.be/software
-   Scikit-learn, indeed: http://scikit-learn.org/

## Dependencies

The only dependencies are scikit-learn, numpy and scipy.

No installation is required.

## Example

Using the SimpleESN class is easy as:

```python
from scikit_esn import SimpleESN
import scipy.io as sio

def load_matsets(dataset_name):
    # Load dataset
    data = sio.loadmat('./dataset/' + dataset_name + '.mat')
    X = data['X']  # shape is [N,T,V]
    # if len(X.shape) < 3:
    #     X = np.atleast_3d(X)
    Y = data['Y']  # shape is [N,1]
    Xte = data['Xte']
    Yte = data['Yte']
    return X, Y, Xte, Yte

X, Y, Xte, Yte = load_matsets('LIB')

# Simple training
# ------ Hyperparameters ------
# Parameters for ESN
hp_n_internal_units = 1000  # size of the reservoir
hp_connectivity = 0.33  # percentage of nonzero connections in the reservoir
hp_weight_scaling = 1.12  # scaling of the reservoir weight
hp_input_scaling = 0.47  # scaling of the input weights
hp_noise_level = 0.07  # noise in the reservoir state update

my_esn = SimpleESN(n_components=hp_n_internal_units,
                   input_scaling=hp_input_scaling,
                   weight_scaling=hp_weight_scaling,
                   connectivity=hp_connectivity,
                   noise_level=hp_noise_level)

my_esn.fit(X,Y)

accuracy = my_esn.accuracy(Xte, Yte)

print('Acc: %.3f' % accuracy)
```

It could also be part of a Pipeline:

```python
from scikit_esn import SimpleESN
# Pick your classifier
pipeline = Pipeline([('esn', SimpleESN(n_readout=1000)),
                     ('svr', svm.SVR())])
parameters = {
    'esn__weight_scaling': [0.5, 1.0],
    'svr__C': [1, 10]
}
grid_search = GridSearchCV(pipeline, parameters)
grid_search.fit(X_train, y_train)
```
