# Preference-Informed Reward Model for Vehicle Suspension Control

## 1. Overview

This project implements and evaluates a preference-informed reward learning system for controlling a simulated vehicle's suspension. The core idea is to learn a latent representation of user preferences from a small number of pairwise comparisons between trajectory samples. This latent preference vector then conditions a reward model, allowing for rapid adaptation of the vehicle's control behavior to align with a specific user's desires (e.g., a preference for a smooth ride vs. sporty handling).

The methodology is heavily based on the concepts outlined in the `project_intent_description.txt` document.

## 2. Core Concepts

*   **Latent User Embedding Model (`q_ψ`)**: A Transformer-based encoder that infers a latent preference vector `z` from a set of trajectory comparisons `D = { (σ_A, σ_B, y) }`. It learns a distribution `q(z|D)`.
*   **Preference-Conditioned Reward Model (`r_φ`)**: A decoder that acts as a reward function. It predicts a scalar reward based on the current state, action, and the inferred latent preference vector `z`, i.e., `r = r_φ(s, a, z)`.
*   **VAE Framework**: The encoder and decoder are trained together as a Variational Autoencoder (VAE) to predict preferences on the training dataset.
*   **Fast Adaptation**: After pre-training, the model can quickly adapt to a new user's preferences by inferring a new `z` from just a few new comparison queries.

## 3. Setup and Installation

All dependencies are managed via Conda.

1.  **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate pi-model
    ```

## 4. Workflow

The project follows a five-step workflow, from raw data generation to the evaluation of the adapted reward function.

### Step 1: Generate Raw Trajectory Data

First, run the vehicle simulation to generate a base set of trajectories. The simulation uses a simple P-controller with varying gains to create diverse behaviors.

```bash
# Example: Generate 500 trajectories using oracle 'A'
python generate_data.py --num-episodes 500 --oracle-name A --dataset-name raw_trajectories_A
```
*This will create a directory `datasets/raw_trajectories_A` filled with `.pkl` files.*

### Step 2: Build the Preference Dataset

Next, process the raw trajectories into a preference dataset. This script uses feature-based clustering (K-Means) to simulate different user groups and generates pairwise preference labels based on cluster-specific scoring functions.

```bash
# Example: Create a preference dataset with 5 clusters and 20,000 preference pairs
python scripts/build_preference_dataset.py --input-dir datasets/raw_trajectories_A --output-path datasets/preference_dataset_A.pkl --num-clusters 5 --num-pairs 20000
```

### Step 3: Pre-train the VAE Model

Train the VAE (the encoder `q_ψ` and decoder `r_φ`) on the preference dataset created in the previous step. Training progress is logged using Weights & Biases.

```bash
# Make sure to log in to wandb first: wandb login
python pref_learn/train.py \
    --dataset_path datasets/preference_dataset_A.pkl \
    --logging.output_dir "logs" \
    --comment "pretrain_suspension_model_A" \
    --seed 42
```
*This will save the trained model (e.g., `model.pt`) inside the `logs/` directory.*

### Step 4: Interactive Adaptation

Simulate an interactive session to adapt the model to a specific preference. This script loads a pre-trained model, intelligently selects trajectory pairs to query for a preference, and updates the latent vector `z`.

```bash
# Example: Run an adaptation loop with 10 queries
python experiments/run_interactive_adaptation.py \
    --vae_model_path "logs/pretrain_suspension_model_A/.../model.pt" \
    --trajectory_path datasets/raw_trajectories_A \
    --num_queries 10
```
*This script will output an `adapted_z.pt` file containing the final adapted latent vector.*

### Step 5: Evaluate the Adapted Reward Function

Finally, use the adapted latent vector `z` to define a new, personalized reward function `r_new(s, a) = r_φ(s, a, z_adapted)`.

```bash
python experiments/evaluate_adaptation.py \
    --vae_model_path "logs/pretrain_suspension_model_A/.../model.pt" \
    --adapted_z_path adapted_z.pt
```
*This script provides the framework for using the newly defined reward function for downstream tasks or further analysis.*

## 5. File Structure

- **`generate_data.py`**: Main script for running the vehicle simulation.
- **`src/`**: Contains core simulation components (`env.py`, `controller.py`, `oracle.py`).
- **`plant.py`, `bump.py`, `vehicle_model.py`**: Defines the physics of the vehicle and road.
- **`*.yaml`**: Configuration files for the simulation (`simulations.yaml`) and oracle behaviors (`oracle_A.yaml`, etc.).
- **`scripts/build_preference_dataset.py`**: Script to process raw trajectories into a preference dataset.
- **`pref_learn/`**: Contains the core machine learning code.
  - **`train.py`**: Script for training the VAE model.
  - **`models/`**: Defines the VAE architecture (`vae.py`) and data loaders (`utils.py`).
- **`experiments/`**: Contains scripts for the adaptation and evaluation phases.
  - **`run_interactive_adaptation.py`**: Simulates the active querying and adaptation process.
  - **`evaluate_adaptation.py`**: Loads the adapted `z` to create a new reward function.
- **`environment.yml`**: Conda environment file listing all dependencies.
- **`project_intent_description.txt`**: The detailed methodological document for this project.
