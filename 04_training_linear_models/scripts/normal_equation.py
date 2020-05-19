theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

theta_best_svd, ... = np.linalg.lstsq(X_b, y, rcond=1e-6)