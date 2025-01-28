import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([1.5, 3.7, 2.9, 5.4, 7.2], dtype=float)

#Calculate the mean of X and y
mean_X = np.mean(X)
mean_y = np.mean(y)

numerator = np.sum((X - mean_X) * (y - mean_y))
denominator = np.sum((X - mean_X) ** 2)
m = numerator / denominator
c = mean_y - m * mean_X

#Predict y values using the regression line
y_pred = m * X + c

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Linear Regression ')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.legend()
plt.grid(True)
plt.show()

# Print the slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")
