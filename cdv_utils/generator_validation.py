"""
RealCause Generator Validation Module

This module provides functions for training and validating RealCause generators
for the sepsis dataset, including statistical tests to ensure the generated
data is indistinguishable from the real data.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from tqdm import tqdm
from models import TarNet, preprocess, TrainingParams, MLPParams
from models import distributions


def setup_model_architectures():
    """
    Define multiple model architectures for exploration.
    
    Returns:
    --------
    list
        List of architecture configurations
    """
    return [
        {
            "name": "medium",
            "n_hidden_layers": 3,
            "dim_h": 128,
            "activation": torch.nn.ReLU(),
            "dropout": 0.2
        }
    ]


def setup_training_parameters(lr=0.0001, batch_size=32, num_epochs=400):
    """
    Set up training parameters for RealCause models.
    
    Parameters:
    -----------
    lr : float
        Learning rate
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of training epochs
    
    Returns:
    --------
    TrainingParams
        Configured training parameters
    """
    return TrainingParams(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        eval_every=10,
        print_every_iters=100
    )


def setup_outcome_distribution(distribution_type="sigmoid_flow", ndim=10):
    """
    Set up the outcome distribution for the RealCause model.
    
    Parameters:
    -----------
    distribution_type : str
        Type of distribution to use
    ndim : int
        Number of dimensions for flow-based distributions
        
    Returns:
    --------
    distribution
        Configured outcome distribution
    """
    if distribution_type == "sigmoid_flow":
        return distributions.SigmoidFlow(ndim=ndim)
    elif distribution_type == "factorial_gaussian":
        return distributions.FactorialGaussian()
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")


def train_realcause_model(w, t, y, architecture, training_params, distribution, 
                         outcome_min=0, outcome_max=1, seed=420, saveroot="save/sepsis_model",
                         w_cols=None):
    """
    Train a single RealCause model with specified architecture.
    
    Parameters:
    -----------
    w, t, y : numpy arrays
        Covariates, treatment, and outcome data
    architecture : dict
        Model architecture configuration
    training_params : TrainingParams
        Training parameters
    distribution : distribution object
        Outcome distribution
    outcome_min, outcome_max : float
        Outcome value bounds
    seed : int
        Random seed
    saveroot : str
        Directory to save models
    w_cols : list
        Covariate column names
        
    Returns:
    --------
    tuple
        (model, metrics, plots, model_path)
    """
    arch_name = architecture["name"]
    model_dir = os.path.join(saveroot, f"{arch_name}_seed_{seed}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nTraining {arch_name} architecture:")
    print(f"- Hidden layers: {architecture['n_hidden_layers']}")
    print(f"- Hidden units: {architecture['dim_h']}")
    print(f"- Activation: {architecture['activation'].__class__.__name__}")
    
    # Set up network parameters
    mlp_params = MLPParams(
        n_hidden_layers=architecture["n_hidden_layers"],
        dim_h=architecture["dim_h"],
        activation=architecture["activation"],
    )
    
    network_params = dict(
        mlp_params_w=mlp_params,
        mlp_params_t_w=mlp_params,
        mlp_params_y0_w=mlp_params,
        mlp_params_y1_w=mlp_params,
    )
    
    # Initialize and train the model
    model = TarNet(
        w, t, y,
        training_params=training_params,
        network_params=network_params,
        binary_treatment=True,
        outcome_distribution=distribution,
        outcome_min=outcome_min,
        outcome_max=outcome_max,
        train_prop=0.5,
        val_prop=0.1,
        test_prop=0.4,
        seed=seed,
        early_stop=True,
        patience=500,
        ignore_w=False,
        grad_norm=float("inf"),
        w_transform=preprocess.Standardize,
        y_transform=preprocess.Normalize,
        savepath=os.path.join(model_dir, 'model.pt')
    )
    
    # Train the model
    model.train()
    
    # Save the model
    torch.save([net.state_dict() for net in model.networks], model.savepath)
    
    # Evaluate the model
    metrics = model.get_univariate_quant_metrics(dataset="test")
    metrics.update(model.get_multivariate_quant_metrics(dataset="test"))
    
    # Calculate ATE
    ate = model.ate().item()
    noisy_ate = model.noisy_ate(seed=seed).item()
    
    # Plot distributions
    plots = model.plot_ty_dists(verbose=False)
    for i, plot in enumerate(plots):
        plot.savefig(os.path.join(model_dir, f"distribution_plot_{i}.png"))
        plt.close(plot)
    
    # Save metrics
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        metrics_json = {k: float(v) for k, v in metrics.items()}
        json.dump({
            "metrics": metrics_json,
            "ate": float(ate),
            "noisy_ate": float(noisy_ate)
        }, f, indent=2)
    
    return model, metrics, plots, model.savepath


def train_multiple_models(w, t, y, w_cols=None, saveroot="save/sepsis_model", seeds=None):
    """
    Train multiple RealCause models with different architectures and seeds.
    
    Parameters:
    -----------
    w, t, y : numpy arrays
        Covariates, treatment, and outcome data
    w_cols : list
        Covariate column names
    saveroot : str
        Directory to save models
    seeds : list
        Random seeds to use
        
    Returns:
    --------
    tuple
        (results, best_model_selector_dict)
    """
    if seeds is None:
        seeds = [420]
    
    # Set up configurations
    architectures = setup_model_architectures()
    training_params = setup_training_parameters()
    distribution = setup_outcome_distribution()
    
    os.makedirs(saveroot, exist_ok=True)
    
    results = {}
    best_model_selector_dict = {}
    
    for arch in architectures:
        arch_name = arch["name"]
        results[arch_name] = {}
        
        for seed in seeds:
            model, metrics, plots, model_path = train_realcause_model(
                w, t, y, arch, training_params, distribution, 
                seed=seed, saveroot=saveroot, w_cols=w_cols
            )
            
            # Store results
            results[arch_name][f"seed_{seed}"] = {
                "metrics": metrics,
                "ate": model.ate().item(),
                "noisy_ate": model.noisy_ate(seed=seed).item(),
                "model_path": model_path
            }
            
            # Keep model for selection
            best_model_selector_dict[arch_name] = deepcopy(model)
            
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save all results
    with open(os.path.join(saveroot, 'all_results.json'), 'w') as f:
        json_results = {}
        for arch, arch_results in results.items():
            json_results[arch] = {}
            for seed, seed_results in arch_results.items():
                json_results[arch][seed] = {
                    "metrics": {k: float(v) for k, v in seed_results["metrics"].items()},
                    "ate": float(seed_results["ate"]),
                    "noisy_ate": float(seed_results["noisy_ate"]),
                    "model_path": seed_results["model_path"]
                }
        json.dump(json_results, f, indent=2)
    
    return results, best_model_selector_dict


def analyze_model_performance(results, saveroot="save/sepsis_model"):
    """
    Analyze the performance of trained models.
    
    Parameters:
    -----------
    results : dict
        Training results from train_multiple_models
    saveroot : str
        Directory where results are saved
        
    Returns:
    --------
    pd.DataFrame
        Performance comparison dataframe
    """
    # Load results if needed
    if isinstance(results, str):
        with open(results, 'r') as f:
            results = json.load(f)
    
    # Create performance comparison dataframe
    performance_data = []
    for arch, arch_results in results.items():
        for seed, seed_results in arch_results.items():
            metrics = seed_results["metrics"]
            performance_data.append({
                "Architecture": arch,
                "Seed": seed,
                "Univariate: Y KS p-value": metrics["y_ks_pval"],
                "Univariate: T KS p-value": metrics["t_ks_pval"],
                "Univariate: Y ES p-value": metrics["y_es_pval"],
                "Univariate: T ES p-value": metrics["t_es_pval"],
                "Univariate: Y Wasserstein": metrics["y_wasserstein1_dist"],
                "Univariate: T Wasserstein": metrics["t_wasserstein1_dist"],
                "Multivariate: Wasserstein1 p-value": metrics["wasserstein1 pval"],
                "Multivariate: Wasserstein2 p-value": metrics["wasserstein2 pval"],
                "Multivariate: kNN p-value": metrics["kNN pval"],
                "Multivariate: Energy p-value": metrics["Energy pval"],
                "Multivariate: Friedman-Rafsky p-value": metrics["Friedman-Rafsky pval"],
                "ATE": seed_results["ate"],
                "Model Path": seed_results["model_path"]
            })
    
    return pd.DataFrame(performance_data)


def select_best_model(results, best_model_selector_dict, criterion="medium"):
    """
    Select the best performing model based on specified criteria.
    
    Parameters:
    -----------
    results : dict
        Training results
    best_model_selector_dict : dict
        Dictionary of trained models
    criterion : str
        Model selection criterion
        
    Returns:
    --------
    model
        Best performing model
    """
    # For now, just return the medium architecture model
    # This can be extended with more sophisticated selection criteria
    return best_model_selector_dict[criterion]


def validate_generator_quality(model, seed=42, statistical_tests=True):
    """
    Validate that the generator produces data indistinguishable from real data.
    
    Parameters:
    -----------
    model : TarNet
        Trained RealCause model
    seed : int
        Random seed for sampling
    statistical_tests : bool
        Whether to run statistical tests
        
    Returns:
    --------
    dict
        Validation results including statistical test p-values
    """
    if statistical_tests:
        # Get statistical metrics
        univariate_metrics = model.get_univariate_quant_metrics(dataset="test")
        multivariate_metrics = model.get_multivariate_quant_metrics(dataset="test")
        
        # Combine all metrics
        validation_results = {}
        validation_results.update(univariate_metrics)
        validation_results.update(multivariate_metrics)
        
        # Add ATE comparison
        validation_results["ate"] = model.ate().item()
        validation_results["noisy_ate"] = model.noisy_ate(seed=seed).item()
        
        return validation_results
    
    return {"message": "Statistical tests not performed"}


def generate_synthetic_data(model, seed=42, dataset='train'):
    """
    Generate synthetic data using the trained RealCause model.
    
    Parameters:
    -----------
    model : TarNet
        Trained RealCause model
    seed : int
        Random seed for generation
    dataset : str
        Dataset type ('train', 'test', 'val')
        
    Returns:
    --------
    tuple
        (w, t, y0, y1) - covariates, treatment, and potential outcomes
    """
    w, t, (y0, y1) = model.sample(
        w=None, transform_w=True, untransform=True,
        seed=seed, dataset=dataset, overlap=1,
        causal_effect_scale=None, deg_hetero=1.0, ret_counterfactuals=True
    )
    
    return w, t, y0, y1


def create_dataframe_from_synthetic_data(w, t, y0, y1, w_cols):
    """
    Create a pandas DataFrame from synthetic data arrays.
    
    Parameters:
    -----------
    w : numpy.array
        Covariates
    t : numpy.array
        Treatment assignments
    y0 : numpy.array
        Control outcomes
    y1 : numpy.array
        Treatment outcomes
    w_cols : list
        Covariate column names
        
    Returns:
    --------
    pd.DataFrame
        Formatted synthetic data
    """
    # Create dataframe with original column names if available
    if w.shape[1] == len(w_cols):
        df = pd.DataFrame(w, columns=w_cols)
    else:
        column_names = [f"feature_{i}" for i in range(w.shape[1])]
        df = pd.DataFrame(w, columns=column_names)
    
    df['t'] = t
    df['y0'] = y0.flatten()
    df['y1'] = y1.flatten()
    df['y'] = np.where(t.flatten() == 0, y0.flatten(), y1.flatten())
    df['ite'] = y1.flatten() - y0.flatten()
    
    return df