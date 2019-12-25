import numpy as np

w = 3
b = 0.5
x_lin = np.linspace(0, 100, 101)
y = (x_lin + np.random.randn(101) * 5) * w + b
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')
plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()

# MAE

def mean_absolute_error(y, yp):
    mae = MAE = abs(y - yp).mean()
    return mae

MAE = mean_absolute_error(y, y_hat)
print("The Mean absolute error is %.3f" % (MAE))

# MSE

def mean_squared_error(y, yp):
    mse = MSE = ((y - yp)**2).mean()
    return mse

MSE = mean_squared_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))