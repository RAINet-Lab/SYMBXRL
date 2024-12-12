'''
The following code is part of "SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks" 
Copyright - RESILIENT AI NETWORK LAB, IMDEA NETWORKS

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''

import numpy as np

class PSquareQuantileApproximator:
    """
    This class applies the P² algorithm for dynamic quantile approximation.
    """
    def __init__(self, p=50):
        self.p = p / 100  # Convert percentile to a fraction for internal use

        self.reset()

    def reset(self):
        """Initialize or reset the quantile approximator."""
        self.n = [0, 1, 2, 3, 4]  # Marker positions
        self.ns = [0, 2 * self.p, 4 * self.p, 2 + 2 * self.p, 4]  # Desired marker positions
        self.dns = [0, self.p / 2, self.p, (1 + self.p) / 2, 1]  # Desired position increments
        self.q = []  # Marker heights

    def fit(self, X):
        """Fit the model to the data."""
        self.reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Incrementally fit the model to the data."""
        for x in X:
            self._partial_fit_single(x)
        return self

    def _partial_fit_single(self, x):
        """Fit the model to a single data point."""
        if len(self.q) < 5:
            self.q.append(x)
            self.q.sort()
            return self
        
        # Determine marker position for the new data point
        if x <= self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[-1]:
            self.q[-1] = x
            k = 4
        else:
            k = next(i for i, q in enumerate(self.q) if x < q)

        # Increment positions and desired positions
        for i in range(k, 5):
            self.n[i] += 1
        
        self.ns = [ns + dns for ns, dns in zip(self.ns, self.dns)]    

        # Adjust marker heights
        for i in range(1, 4):
            d = self.ns[i] - self.n[i]
            if (d >= 1 and self.n[i+1] - self.n[i] > 1) or (d <= -1 and self.n[i-1] - self.n[i] < -1):
                d_sign = np.sign(d)
                q_para = self._parabolic(i, d_sign)
                self.q[i] = q_para if self.q[i-1] < q_para < self.q[i+1] else self._linear(i, d_sign)
                self.n[i] += d_sign

    def _parabolic(self, i, d):
        """Calculate parabolic prediction for marker height adjustment."""
        i = int(i)
        d = int(d)
        return self.q[i] + (d * (self.n[i] - self.n[i-1] + d) * (self.q[i+1] - self.q[i]) / (self.n[i+1] - self.n[i])
                            + d * (self.n[i+1] - self.n[i] - d) * (self.q[i] - self.q[i-1]) / (self.n[i] - self.n[i-1])) / (self.n[i+1] - self.n[i-1])

    def _linear(self, i, d):
        """Calculate linear prediction for marker height adjustment."""
        i = int(i)
        d = int(d)
        return self.q[i] + d * (self.q[i+d] - self.q[i]) / (self.n[i+d] - self.n[i])

    def score(self):
        # Check if self.q is empty
        if not self.q:
            # Return a sensible default or raise an error
            # For example, return None or 0, or raise an informative error
            return None  # or return 0, or raise an Exception

        # If self.p is 0 or 1, handle these edge cases
        if self.p == 0:
            return self.q[0]
        if self.p == 1:
            return self.q[-1]  # Use -1 to access the last element

        # If len(self.q) < 5, use NumPy's percentile
        # This condition will now only be reached if self.q is not empty
        if len(self.q) < 5:
            return np.percentile(self.q, self.p * 100)

        # For the general case when there are 5 or more data points
        return self.q[2]  # Assuming this is how you wish to calculate the score generally
    
    def get_markers(self):
        return self.q
