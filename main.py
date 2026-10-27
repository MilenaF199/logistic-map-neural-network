# Proyect: Neuronal network for dynamic systems


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Logistic map
def logistic_map(r, x0, n_steps):
    x = np.zeros(n_steps + 1)
    x[0] = x0

    for n in range(n_steps):
        x[n + 1] = r * x[n] * (1 - x[n])

    return x


# Parameters
x0 = 0.2
n_steps = 100

# Periodic and chaotic regimes
periodic = logistic_map(r=3.2, x0=x0, n_steps=n_steps)
chaotic = logistic_map(r=4.0, x0=x0, n_steps=n_steps)

print("Periodic trajectory:")
print(periodic[:10])

print("\nChaotic trajectory:")
print(chaotic[:10])



# Plot both trajectories
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

n = np.arange(n_steps + 1)

axes[0].plot(n, periodic, linewidth=1.5)
axes[0].set_title("Periodic regime (r = 3.2)")
axes[0].set_xlabel("Iteration n")
axes[0].set_ylabel("x_n")
axes[0].grid(alpha=0.3)

axes[1].plot(n, chaotic, linewidth=1.5)
axes[1].set_title("Chaotic regime (r = 4.0)")
axes[1].set_xlabel("Iteration n")
axes[1].set_ylabel("x_n")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("logistic_map_trajectories.png", dpi=300)
plt.show()


# Dataset 
def generate_dataset(r_values, n_trajectories=100, n_steps=50, seed=42):
    rng = np.random.default_rng(seed)

    X = []
    y = []
    groups = []

    trajectory_id = 0

    for r in r_values:
        for _ in range(n_trajectories):
            x0 = rng.uniform(0.05, 0.95)
            trajectory = logistic_map(r, x0, n_steps)

            for n in range(n_steps):
                X.append([trajectory[n], r])
                y.append(trajectory[n + 1])
                groups.append(trajectory_id)

            trajectory_id += 1

    return np.array(X), np.array(y), np.array(groups)


X, y, groups = generate_dataset(
    r_values=[3.2, 4.0],
    n_trajectories=100,
    n_steps=50
)

# Split by trajectory, not by individual observations
unique_groups = np.unique(groups)

train_groups, test_groups = train_test_split(
    unique_groups,
    test_size=0.2,
    random_state=42
)

train_mask = np.isin(groups, train_groups)
test_mask = np.isin(groups, test_groups)

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# MLP model
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="tanh",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True
    )
)

model.fit(X_train, y_train)

# One-step predictions
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nNeural Network Results")
print("MAE:", mae)
print("RMSE:", rmse)


#Baseline
# Persistence baseline
baseline_pred = X_test[:, 0]

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_pred)
)

results = pd.DataFrame({
    "Model": ["Persistence Baseline", "Neural Network"],
    "MAE": [baseline_mae, mae],
    "RMSE": [baseline_rmse, rmse]
})

print("\nModel Comparison")
print(results)

results.to_csv("preliminary_results.csv", index=False)



# Multiple steps prediction
def neural_trajectory(model, r, x0, n_steps):
    predictions = np.zeros(n_steps + 1)
    predictions[0] = x0

    for n in range(n_steps):
        input_data = np.array([[predictions[n], r]])
        next_value = model.predict(input_data)[0]

        predictions[n + 1] = next_value

    return predictions


# Compare exact and predicted trajectories
horizon = 30

for r in [3.2, 4.0]:
    exact = logistic_map(r, x0=0.23, n_steps=horizon)

    predicted = neural_trajectory(
        model, r, x0=0.23, n_steps=horizon
    )

    plt.figure(figsize=(9, 4))
    plt.plot(exact, label="Exact trajectory", linewidth=2)
    plt.plot(predicted, "--", label="Neural network")

    plt.title(f"Multi-step prediction (r = {r})")
    plt.xlabel("Prediction step")
    plt.ylabel("State")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"prediction_r_{r}.png", dpi=300)
    plt.show()

    print(f"\nr = {r}")
    print("Multi-step MAE:",
          mean_absolute_error(exact[1:], predicted[1:]))
    print("Multi-step RMSE:",
          np.sqrt(mean_squared_error(exact[1:], predicted[1:])))




# Evaluate the experiment with mutiple trajectories
# Robust multi-step evaluation

rng = np.random.default_rng(123)

initial_conditions = rng.uniform(0.05, 0.95, 30)
horizons = [1, 5, 10, 20, 30]

evaluation_results = []

for r in [3.2, 4.0]:
    for horizon in horizons:
        errors = []

        for x0 in initial_conditions:
            exact = logistic_map(r, x0, horizon)

            predicted = neural_trajectory(
                model, r, x0, horizon
            )

            # Error at the selected prediction horizon
            errors.append(
                abs(exact[horizon] - predicted[horizon])
            )

        evaluation_results.append({
            "Regime": "Periodic" if r == 3.2 else "Chaotic",
            "r": r,
            "Horizon": horizon,
            "MAE": np.mean(errors),
            "RMSE": np.sqrt(np.mean(np.square(errors)))
        })

evaluation_df = pd.DataFrame(evaluation_results)

print("\nMulti-step evaluation:")
print(evaluation_df.to_string(index=False))

evaluation_df.to_csv(
    "multi_step_evaluation.csv",
    index=False
)

# Plot error growth
plt.figure(figsize=(9, 5))

for regime in ["Periodic", "Chaotic"]:
    subset = evaluation_df[
        evaluation_df["Regime"] == regime
    ]

    plt.plot(
        subset["Horizon"],
        subset["MAE"],
        marker="o",
        label=regime
    )

plt.xlabel("Prediction horizon")
plt.ylabel("Mean Absolute Error")
plt.title("Prediction Error vs Forecast Horizon")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "error_vs_horizon.png",
    dpi=300
)

plt.show()
