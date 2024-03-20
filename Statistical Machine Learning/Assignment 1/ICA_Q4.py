#Q4
import numpy as np
import matplotlib.pyplot as plt
import math
class ICA:
    def centering(x):
        X_centered = mixed_x - np.mean(x, axis=0)
        return X_centered
    def covariance(y):
        cov_y = np.cov(y)
        return cov_y
    def eigenvalues(A):
        l = np.linalg.eigvals(A)
        return l
    def center(X):
        X = np.array(X)
        mean = X.mean(axis=1, keepdims=True)
        return X- mean
    def whitening(X):
        cov = np.cov(X)
        d, E = np.linalg.eigh(cov)
        D = np.diag(d)
        D_inv = np.sqrt(np.linalg.inv(D))
        X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
        return X_whiten
    def g(x):
        return np.tanh(x)
    def g_der(x):
        return 1 - ICA.g(x) * ICA.g(x)
    def calculate_new_w(w, X):
        w_new = (X * ICA.g(np.dot(w.T, X))).mean(axis=1) - ICA.g_der(np.dot(w.T, X)).mean() * w
        w_new /= np.sqrt((w_new ** 2).sum())
        return w_new
    def ica(X, iterations, tolerance=1e-4):
        X = ICA.center(X)
        X = ICA.whitening(X)
        print('Covariance Matrix of Whitened Matrix as Identity Matrix:-\n',np.cov(X))
        components_nr = X.shape[0]
        W = np.zeros((components_nr, components_nr), dtype=X.dtype)
        for i in range(components_nr):
            w = np.random.rand(components_nr)
            for j in range(iterations):
                w_new = ICA.calculate_new_w(w, X)
                if i >= 1:
                    w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
                
                distance = np.abs(np.abs((w * w_new).sum()) - 1)
                w = w_new
                if distance < tolerance:
                    break    
            W[i, :] = w
        S = np.dot(W, X)
        return S
t = np.linspace(0,2*np.pi,1000)
x1 = np.sin(2*np.pi*t)#sinusoidal signal
#print(x1)
x2 = t%1 #ramp signal
plt.title("Original signals")
plt.plot(t,x1,label = "Sinusoidal signal")
plt.plot(t,x2,label = "Ramp signal")
plt.legend()
plt.grid()
plt.show()
A = np.array([[0.5, 1], [1, 0.5]])
signals = np.vstack((x1, x2)).T
print('Signal Matrix:-\n',signals)
mixed_x = np.dot(signals,A)
print('Mixed Signals matrix:-\n',mixed_x)
plt.title("Mixed signals")
plt.plot(t, mixed_x.T[0],label = "Mixed signal1")
plt.plot(t, mixed_x.T[1],label = "Mixed signal2")
plt.legend()
plt.grid()
plt.show()

S = ICA.ica(mixed_x.T,1000)
plt.title("Separated Signals")
plt.plot(t,S[0],label="Ramp Signal")
plt.plot(t,S[1],label="Sinusoidal signal")
plt.legend()
plt.grid()
plt.show()

