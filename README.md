# logistic-map-neural-network
Neural network forecasting of periodic and chaotic dynamics in the logistic map.


# Predicting Periodic and Chaotic Dynamics Using Neural Networks

This project investigates the multi-step forecasting performance of a multilayer perceptron (MLP) in periodic and chaotic regimes of the logistic map.

## Mathematical Model

The logistic map is defined as:

\(x_{n+1} = r x_n(1-x_n)\)

Two parameter values are considered:

- **r = 3.2:** Periodic regime.
- **r = 4.0:** Chaotic regime.

## Methodology

Synthetic trajectories are generated using the logistic map. A multilayer perceptron with two hidden layers of 32 neurons is trained to predict the next state from the current state and control parameter.

A persistence baseline is used for one-step prediction.

Recursive predictions are evaluated at horizons of 1, 5, 10, 20, and 30 steps using mean absolute error (MAE) and root mean squared error (RMSE).

## Preliminary Results

The MLP achieves a one-step MAE of 0.003844, compared with 0.341258 for the persistence baseline.

At a horizon of 30 steps, the MAE is 0.010647 for the periodic regime and 0.342372 for the chaotic regime.

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Execute:

```bash
python main.py
```

The script generates the synthetic dataset, trains the model, evaluates forecasting performance, and produces the experimental results.

## Reproducibility

The experiments use fixed random seeds for dataset generation, model training, and multi-step evaluation.
