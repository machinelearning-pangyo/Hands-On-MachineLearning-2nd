from sklearn.preprocessing import PolynomialFeatures
poly_ = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_.fit_trasform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)