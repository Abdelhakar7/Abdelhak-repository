import numpy as np
import pandas as pd
import math

# Read the CSV file using a raw string for the path
df = pd.read_csv(r"C:\Users\abdelhak pc\Downloads\test_scores.csv")

# Print the first few rows of the DataFrame to verify it's loaded correctly
print(df.head())

# Assign the math column to x and cs column to y as numpy arrays
x = df['math'].values
y = df['cs'].values

print("x:", x)
print("y:", y)


def gradient_descent(x, y, learning_rate=0.1, iterations=1000, tolerance=1e-20):
    m_curr = b_curr = 0
    n = len(x)
    previous_cost = float('inf')

    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        current_cost = (1 / n) * np.sum((y - y_pred) ** 2)

        if math.isclose(previous_cost, current_cost, rel_tol=tolerance):
            print(f"Stopping early at iteration {i} due to minimal cost change")
            break

        previous_cost = current_cost

        md = -(2 / n) * np.sum(x * (y - y_pred))
        bd = -(2 / n) * np.sum(y - y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if i % 100 == 0:  # Print every 100 iterations for brevity
            print(f"m {m_curr}, b {b_curr}, cost {current_cost}, iteration {i}")

    return m_curr, b_curr


# Perform gradient descent with a lower learning rate
m, b = gradient_descent(x, y, learning_rate=0.0001)
print(f"Final values: m = {m}, b = {b}")

# Calculate the correlation between math and cs scores
correlation = df['math'].corr(df['cs'])
print(f"The correlation between math scores and cs scores is: {correlation}")

