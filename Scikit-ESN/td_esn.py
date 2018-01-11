"""Simple Echo State Network
"""

# Copyright (C) 2015 Sylvain Chevallier <sylvain.chevallier@uvsq.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array
from numpy import zeros, ones, concatenate, array, tanh, vstack, arange
import numpy as np
import scipy.linalg as la
from scipy import sparse
from scipy.sparse import linalg as slinalg


class SimpleESN(BaseEstimator, TransformerMixin):
    """Simple Echo State Network (ESN)

    Neuron reservoir of sigmoid units, with recurrent connection and random
    weights. Forget factor (or input_scaling) ensure echoes in the network. No
    learning takes place in the reservoir, readout is left at the user's
    convience. The input processed by these ESN should be normalized in [-1, 1]

    Parameters
    ----------
    n_readout : int
        Number of readout neurons, chosen randomly in the reservoir. Determines
        the dimension of the ESN output.

    n_components : int, optional
        Number of neurons in the reservoir, 100 by default.

    input_scaling : float, optional
        input_scaling (forget) factor for echoes, strong impact on the dynamic of the
        reservoir. Possible values between 0 and 1, default is 0.5

    weight_scaling : float, optional
        Spectral radius of the reservoir, i.e. maximum eigenvalue of the weight
        matrix, also strong impact on the dynamical properties of the reservoir.
        Classical regimes involve values around 1, default is 0.9

    discard_steps : int, optional 即丢弃的时间序列的位数
        Discard first steps of the timeserie, to allow initialisation of the
        network dynamics.

    random_state : integer or numpy.RandomState, optional 即设定种子，如果为空，则随机生成一个种子。
        Random number generator instance. If integer, fixes the seed.
    
    connectivity : number of nonzero connections in the reservoir

    noise_level  : deviation of the Gaussian noise 

    Attributes
    ----------
    input_weights_ : array_like, shape (n_features,)
        Weight of the input units

    weights_       : array_Like, shape (n_components, n_components)
        Weight matrix for the reservoir

    state_matrix   : array_like, shape(n_samples, n_timeseries, n_components)
        state matrix for the reservoir

    last_state     : array_like, shape(n_samples, 1, n_components)
        laste state matrix for the reservoir

    components_ : array_like, shape (n_samples, 1+n_features+n_components)
        Activation of the n_components reservoir neurons, including the
        n_features input neurons and the bias neuron, which has a constant
        activation.

    # readout_idx_ : array_like, shape (n_readout,)
    #     Index of the randomly selected readout neurons

    Example
    -------

    >>> from simple_esn import SimpleESN
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> X = np.random.randn(n_samples, n_features)
    >>> esn =SimpleESN(n_readout = 2)
    >>> echoes = esn.fit_transform(X)
    """

    def __init__(self, n_components=100, input_scaling=0.5,
                 weight_scaling=0.9,connectivity=0.3,noise_level=0.01,discard_steps=0,random_state=None):
        self.n_components = n_components
        self.input_scaling = input_scaling
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = check_random_state(random_state)
        self.connectivity = connectivity
        self.noise_level=noise_level
        self.input_weights_ = None
        # self.weights_ = None
        self.weights_=self._initialize_internal_weights(self.n_components,self.connectivity,self.weight_scaling)
        self.state_matrix=None
        self.last_state=None

    def _initialize_internal_weights(self,n_components,
                                        connectivity, weight_scaling):
            # The eigs function might not converge. Attempt until it does.
            convergence = False
            while (not convergence):
                # Generate sparse, uniformly distributed weights.
                internal_weights = sparse.rand(n_components,
                                            n_components,
                                            density=connectivity).todense()

                # Ensure that the nonzero values are
                # uniformly distributed in [-0.5, 0.5]
                internal_weights[np.where(internal_weights > 0)] -= 0.5

                try:
                    # Get the largest eigenvalue which means spectral_radius
                    spectral_radius, _ = slinalg.eigs(internal_weights, k=1, which='LM')

                    convergence = True

                except:
                    continue

            # Adjust the weight_scaling
            internal_weights /= np.abs(spectral_radius) / weight_scaling
            return internal_weights

    def _fit_transform(self, X):
        # n_samples, n_features = X.shape
        # X = check_array(X, ensure_2d=True)
        
        # span to 3-dim
        if len(X.shape) < 3:
            X = np.atleast_3d(X)
        n_samples, n_timeseries, n_features = X.shape

        if self.input_weights_ is None:
            self.input_weights_ = 2 * self.random_state.rand(self.n_components, n_features) - 1.0

        self.state_matrix=np.empty((n_samples, n_timeseries - self.discard_steps, self.n_components),dtype=float)

        previous_state = np.zeros((n_samples, self.n_components), dtype=float)

        for t in range(n_timeseries):
            current_input = X[:, t, :] * self.input_scaling

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = self.weights_.dot(previous_state.T) + self.input_weights_.dot(current_input.T)

            state_before_tanh += np.random.rand(self.n_components, n_samples) * self.noise_level

            previous_state = np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > self.discard_steps - 1):
                self.state_matrix[:, t - self.discard_steps , :] = previous_state

        self.last_state=self.state_matrix[:,-1,:]

        return self

    def fit(self, X, y=None):
        """Initialize the network

        This is more compatibility step, as no learning takes place in the
        reservoir.

        Parameters
        ----------
        X : array-like shape, (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        self : returns an instance of self.
        """
        self = self._fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """Generate echoes from the reservoir.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.

        Returns
        -------
        readout : array, shape (n_samples, n_readout)
            Reservoir activation generated by the readout neurons
        """
        self = self._fit_transform(X)
        return self.last_state

    def transform(self, X):
       # n_samples, n_features = X.shape
        # X = check_array(X, ensure_2d=True)
        
        # span to 3-dim
        if len(X.shape) < 3:
            X = np.atleast_3d(X)
        n_samples, n_timeseries, n_features = X.shape

        if self.input_weights_ is None:
            self.input_weights_ = 2 * self.random_state.rand(self.n_components, n_features) - 1.0

        self.state_matrix=np.empty((n_samples, n_timeseries - self.discard_steps, self.n_components),dtype=float)

        previous_state = np.zeros((n_samples, self.n_components), dtype=float)

        for t in range(n_timeseries):
            current_input = X[:, t, :] * self.input_scaling

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = self.weights_.dot(previous_state.T) + self.input_weights_.dot(current_input.T)

            state_before_tanh += np.random.rand(self.n_components, n_samples) * self.noise_level

            previous_state = np.tanh(state_before_tanh).T

            # Store everything after the dropout period
            if (t > self.discard_steps - 1):
                self.state_matrix[:, t - self.discard_steps , :] = previous_state

        self.last_state=self.state_matrix[:,-1,:]
        return self.last_state
