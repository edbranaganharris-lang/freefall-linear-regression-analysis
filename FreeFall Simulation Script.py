
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math

# ======================== Reading and Preprocessing the CSV Data ========================
file_path = r"C:\Users\ed\OneDrive - University of Bristol\Documents\Github\exercise3data.csv"
data = pd.read_csv(file_path, comment='#', header=None, on_bad_lines="skip")

# Define column names and clean the data
data.columns = ["Material", "Density", "Radius", "Mass", "Temperature", "Pressure", "Height", "Time"]

# Remove rows with missing values and force numerical conversion where needed
data = data.dropna()
for col in ["Density", "Radius", "Mass", "Temperature", "Pressure", "Height", "Time"]:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=["Density", "Radius", "Mass", "Temperature", "Pressure", "Height", "Time"])

# Only keep rows with positive values and known materials
valid_materials = ["iron", "titanium", "magnesium", "silicon_carbide", "zinc_oxide", "silica", "polycarbonate"]
data = data[(data["Density"] > 0) & (data["Radius"] > 0) & (data["Mass"] > 0) &
            (data["Temperature"] > 0) & (data["Pressure"] > 0) & (data["Height"] > 0) &
            (data["Time"] > 0) &
            (data["Material"].isin(valid_materials))]

# ======================== Define Functions for Each Menu Option ========================

def option_data_summary():
    """Option 1: Print the min and max values for each numerical feature."""
    print("\nData Summary (Min/Max values):")
    # Skip the first column ("Material") which is categorical
    for col in data.columns[1:]:
        print(f"{col}: Min = {data[col].min()}, Max = {data[col].max()}")

def option_freefall_and_corr():
    """Option 2: Plot Freefall Time vs Drop Height and display the correlation matrix."""
    # Plot: Freefall Time vs Drop Height
    plt.figure(figsize=(8, 6))
    for material in data["Material"].unique():
        subset = data[data["Material"] == material]
        plt.scatter(subset["Height"], subset["Time"], label=material, alpha=0.6)
    plt.xlabel("Drop Height (m)")
    plt.ylabel("Fall Time (s)")
    plt.title("Freefall Time vs Drop Height")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot: Correlation Matrix Heatmap
    correlation_matrix = data[["Density", "Radius", "Mass", "Temperature", "Pressure", "Height", "Time"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="Spectral", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Freefall Data")
    plt.show()
    
    
    # Option 3: Visualization of Linear vs. Huber Regression
def plot_regression_comparison():
    # Choose feature for visualization
    feature_name = "Height"  # You can also try "Radius" or other features
    X_feature = data[feature_name].values.reshape(-1, 1)  # Reshape for single feature regression
    y = data["Time"].values
    
    # Create and fit Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_feature, y)
    y_pred_linear = linear_model.predict(X_feature)
    
    # Create and fit Huber Regression model
    huber_model = HuberRegressor()
    huber_model.fit(X_feature, y)
    y_pred_huber = huber_model.predict(X_feature)
    
    # Plot the results
    plt.figure(figsize=(10, 6))

    # Scatter plot of the data points
    plt.scatter(X_feature, y, color='blue', label="Data", alpha=0.6)

    # Linear regression line
    plt.plot(X_feature, y_pred_linear, color='red', label="Linear Regression", linewidth=2)

    # Huber regression line
    plt.plot(X_feature, y_pred_huber, color='green', label="Huber Regression", linewidth=2)

    # Adding labels and title
    plt.xlabel(feature_name)
    plt.ylabel("Time (s)")
    plt.title(f"Comparison of Linear Regression vs Huber Regression for {feature_name}")
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def option_regression_analysis():
    """Option 3: Run a standard linear regression and print the beta values (and Huber regression coefficients)."""
    # Prepare the features and target (using unscaled data)
    X = data[["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = data["Time"]

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_scaled, y)
    beta_0 = lin_model.intercept_
    betas = lin_model.coef_
    print("\nLinear Regression Model Coefficients:")
    print(f"Intercept (β0): {beta_0}")
    coef_labels = ["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]
    for label, coef in zip(coef_labels, betas):
        print(f"β{label}: {coef}")

    # Huber Regression (robust to outliers)
    huber_model = HuberRegressor(epsilon=1.35)
    huber_model.fit(X_scaled, y)
    print("\nHuber Regression Coefficients:")
    print(huber_model.coef_)


    
def option_gradient_descent():
    """Option 4: Perform Batch, Stochastic, and Mini-Batch Gradient Descent."""
    # Prepare and scale the features
    scaler = StandardScaler()
    X_features = data[["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]].values
    y_values = data["Time"].values
    X_scaled = scaler.fit_transform(X_features)
    # Add an intercept term (column of ones)
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    m, n = X_scaled.shape
    coef_labels = ["Intercept", "Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]

    #==== Batch Gradient Descent =====
    alpha = 0.01  # Learning rate
    n_iter = 1000  # Number of iterations
    beta = np.zeros(n)
    for _ in range(n_iter):
        gradients = (2/m) * X_scaled.T.dot(X_scaled.dot(beta) - y_values)
        beta -= alpha * gradients
    print("\nBatch Gradient Descent Coefficients:")
    for label, coef in zip(coef_labels, beta):
        print(f"{label}: {coef}")

    #==== Stochastic Gradient Descent ====
    alpha_sgd = 0.001  # Learning rate for SGD
    n_iter_sgd = 100  # Number of iterations for SGD
    beta_sgd = np.zeros(n)
    for _ in range(n_iter_sgd):
        for i in range(m):
            xi = X_scaled[i, :].reshape(1, -1)
            yi = y_values[i]
            gradients = 2 * xi.T.dot(xi.dot(beta_sgd) - yi)
            beta_sgd -= alpha_sgd * gradients
    print("\nStochastic Gradient Descent Coefficients:")
    for label, coef in zip(coef_labels, beta_sgd):
        print(f"{label}: {coef}")

    #=== Mini-Batch Gradient Descent ====
    beta_mbgd = np.zeros(n)
    batch_size = 16
    for _ in range(n_iter):
        indices = np.random.permutation(m)
        X_shuffled = X_scaled[indices]
        y_shuffled = y_values[indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            gradients = (2/len(X_batch)) * X_batch.T.dot(X_batch.dot(beta_mbgd) - y_batch)
            beta_mbgd -= alpha * gradients
    print("\nMini-Batch Gradient Descent Coefficients:")
    for label, coef in zip(coef_labels, beta_mbgd):
        print(f"{label}: {coef}")

def option_train_test_analysis():
    """Option 5: Split the data into training and test sets, train a linear model, and analyze residuals."""
    # Prepare and scale the features
    scaler = StandardScaler()
    X_features = data[["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]].values
    X_scaled = scaler.fit_transform(X_features)
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    
    # Split data into training and test sets while retaining original indices for plotting
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, data["Time"], data.index, test_size=0.1, random_state=7)
    
    # Train the linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    print("\nTrain-Test Analysis (Part 4a):")
    print("Mean residual:", np.mean(residuals))
    print("Standard deviation of residuals:", np.std(residuals))

    # Plot histogram of residuals
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Residuals on Test Data")
    plt.show()

    # Plot residuals vs. Height (using original data indices)
    height_test = data.loc[idx_test, "Height"]
    plt.figure(figsize=(8, 4))
    plt.scatter(height_test, residuals, alpha=0.7)
    plt.xlabel("Height (m)")
    plt.ylabel("Residual")
    plt.title("Residuals vs. Height")
    plt.grid(True)
    plt.show()

def option_analytical_extrapolation():
    """Option 6: Analytical Model Extrapolation (Part 4b).
    
    Compare the analytical predictions (using quadratic drag) with the linear model’s extrapolated predictions
    for heights of 500 m and 1000 m.
    """
    #==== Analytical Model ===
    # Compute average mass and radius from the dataset
    m_avg = data["Mass"].mean()
    r_avg = data["Radius"].mean()
    # For a sphere, the cross-sectional area A = π * r^2
    A_avg = np.pi * (r_avg ** 2)
    
    # Using standard atmospheric conditions
    p_standard = 101325   # Pa
    T_standard = 288.15   # K (15°C)
    M_air = 0.0289652     # kg/mol
    R_const = 8.314462    # J/(K·mol)
    
    # Compute air density: ρ0 = p * M_air / (R * T)
    rho0 = p_standard * M_air / (R_const * T_standard)
    
    # Drag coefficient and constant k = (Cd * ρ0 * A) / 2
    Cd = 0.47
    k_drag = (Cd * rho0 * A_avg) / 2
    
    def analytical_fall_time(h, m, k, g=9.81):
        # Calculate the argument for cosh⁻¹
        argument = math.exp((h * k) / m)
        # cosh⁻¹(x) = ln(x + sqrt(x^2 - 1))
        # Ensure the term under the square root is non-negative.
        if argument**2 - 1 < 0:
            cosh_inv = 0
        else:
            cosh_inv = math.log(argument + math.sqrt(argument**2 - 1))
        t_val = math.sqrt(m / (k * g)) * cosh_inv
        return t_val
    
    h_values = [500, 1000]
    print("\nAnalytical Model Predictions:")
    for h in h_values:
        t_val = analytical_fall_time(h, m_avg, k_drag)
        print(f"At h = {h} m, Analytical Fall Time = {t_val:.2f} s")
    
    #=== Linear Model Extrapolation ====
    # Train a linear model using unscaled data
    X = data[["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]]
    y = data["Time"]
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    
    # Define the feature names (must match the ones used to train the model)
    feature_names = ["Density", "Radius", "Mass", "Temperature", "Pressure", "Height"]

    # Calculate the average values for all features except 'Height'
    features_avg = data[["Density", "Radius", "Mass", "Temperature", "Pressure"]].mean().values

    # Create a new DataFrame for the prediction at h = 500 m

    new_data_500 = pd.DataFrame([list(features_avg) + [500]], columns=feature_names)
    lin_pred_500 = lin_model.predict(new_data_500)[0]

    # And similarly for h = 1000 m:
    new_data_1000 = pd.DataFrame([list(features_avg) + [1000]], columns=feature_names)
    lin_pred_1000 = lin_model.predict(new_data_1000)[0]

    print(f"At h = 500 m, Linear Model Fall Time = {lin_pred_500:.2f} s")
    print(f"At h = 1000 m, Linear Model Fall Time = {lin_pred_1000:.2f} s")


def main_menu():
    while True:
        print("\nChoose an option:")
        print("1. Data Summary (Min/Max)")
        print("2. Freefall Time vs Drop Height & Correlation Matrix")
        print("3. Linear vs Huber Regression Comparison")
        print("4. Regression Analysis Beta Values")
        print("5. Gradient Descent Approaches")
        print("6. Part 4a")
        print("7. Exit")

        option = input("Enter option number: ")

        if option == '1':
            option_data_summary()
        elif option == '2':
            option_freefall_and_corr()
        elif option == '3':
            plot_regression_comparison()  # Call the plotting function for option 3
        elif option == '4':
            option_regression_analysis()
        elif option == '5':
            option_train_test_analysis()
        elif option == '6':
            option_analytical_extrapolation()
        elif option == '7':
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    main_menu()

