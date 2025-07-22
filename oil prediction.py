import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Drilling_Hours': [5, 8, 12, 15, 20, 25, 30],
    'Oil_Output_Barrels': [50, 80, 120, 150, 200, 250, 300]
}
df = pd.DataFrame(data)


X = df[['Drilling_Hours']]
y = df['Oil_Output_Barrels']


model = LinearRegression()
model.fit(X, y)


predicted = model.predict(X)


error = mean_squared_error(y, predicted)
print(f"Prediction Error (MSE): {error:.6f}")


plt.scatter(X, y, color='blue', label='Actual Output')
plt.plot(X, predicted, color='red', label='Predicted Output')
plt.title('Oil Output Prediction Based on Drilling Hours')
plt.xlabel('Drilling Hours')
plt.ylabel('Oil Output (Barrels)')
plt.legend()
plt.show()
