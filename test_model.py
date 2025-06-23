"""
Flux Enhancement Model Testing Suite
===================================
This module provides comprehensive testing capabilities for the trained
flux enhancement detection model, including storm event analysis and
real-time prediction simulation.

Author: [Your Name]
Date: [Current Date]
"""

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from flux_enhancement_nn import FlexibleNN


def load_trained_model():
    """
    Load the trained model and feature scaler from disk.
    
    Returns:
        tuple: (model, scaler) - The trained neural network and StandardScaler
    """
    # Initialize model architecture
    model = FlexibleNN(
        input_size=23,
        hidden_sizes=[128, 64, 32],
        output_size=2,
        dropout_rate=0.3
    )
    
    # Load saved weights
    model.load_state_dict(torch.load('flux_enhancement_model.pth'))
    model.eval()  # Set to evaluation mode
    
    # Load feature scaler
    scaler = joblib.load('flux_enhancement_scaler.pkl')
    
    return model, scaler


def test_on_specific_storm(storm_date='2015-03-17'):
    """
    Test model performance on a specific storm event.
    
    Parameters:
        storm_date (str): Date of the storm event in YYYY-MM-DD format
    """
    print(f"Loading data for storm event analysis: {storm_date}")
    
    # Load labeled data
    df = pd.read_csv('flux_enhancement_labeled_fixed.csv', 
                     index_col=0, parse_dates=True)
    
    # Select data around the storm (20-day window)
    storm_start = pd.to_datetime(storm_date)
    storm_end = storm_start + pd.Timedelta(days=20)
    
    storm_data = df[(df.index >= storm_start) & (df.index <= storm_end)]
    
    if len(storm_data) == 0:
        print(f"ERROR: No data found for storm date {storm_date}")
        print(f"Available date range: {df.index.min()} to {df.index.max()}")
        return
    
    # Load trained model
    model, scaler = load_trained_model()
    
    # Define feature columns used during training
    feature_cols = [
        'flux_filtered', 'lstar_filtered', 'MLT', 'alpha_local',
        'omni_SYM_H', 'omni_B', 'omni_V', 'omni_n',
        'hour', 'day_of_year',
        'flux_rolling_mean_24h', 'flux_rolling_std_24h', 'flux_rolling_max_24h',
        'symh_rolling_mean_24h', 'symh_rolling_min_24h',
        'flux_lag_6h', 'flux_lag_12h', 'flux_lag_24h',
        'symh_lag_6h', 'symh_lag_12h',
        'flux_change_6h', 'flux_change_12h',
        'hours_since_storm'
    ]
    
    # Verify feature availability
    available_features = [col for col in feature_cols if col in storm_data.columns]
    print(f"Using {len(available_features)} features for prediction")
    
    # Extract features and labels
    X = storm_data[available_features].values
    y_true = storm_data['enhancement_label'].values if 'enhancement_label' in storm_data.columns else None
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Scale features and make predictions
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1).numpy()
    
    # Generate visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Particle flux
    axes[0].plot(storm_data.index, storm_data['flux_filtered'], 'b-', alpha=0.7)
    axes[0].set_ylabel('Particle Flux (7.7 MeV)')
    axes[0].set_title(f'Storm Event Analysis: {storm_date}')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Geomagnetic index
    axes[1].plot(storm_data.index, storm_data['omni_SYM_H'], 'g-')
    axes[1].axhline(y=-50, color='r', linestyle='--', label='Storm threshold')
    axes[1].set_ylabel('SYM/H Index (nT)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Model predictions
    if y_true is not None:
        axes[2].scatter(storm_data.index[y_true == 1], 
                       np.ones(sum(y_true == 1)) * 0.8, 
                       color='green', s=20, label='True Enhancement', alpha=0.7)
    
    axes[2].scatter(storm_data.index[predictions == 1], 
                   np.ones(sum(predictions == 1)) * 0.2, 
                   color='red', s=20, label='Predicted Enhancement', alpha=0.7)
    
    # Add enhancement probability
    enhancement_prob = probs[:, 1].numpy()
    axes[2].fill_between(storm_data.index, 0, enhancement_prob, 
                        alpha=0.3, color='orange', label='Enhancement Probability')
    
    axes[2].set_ylabel('Enhancement Status')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'storm_analysis_{storm_date}.png', dpi=150)
    plt.show()
    
    # Print analysis results
    print(f"\nStorm Event Analysis Results ({storm_date}):")
    print(f"Total data points: {len(storm_data)}")
    print(f"Predicted enhancements: {sum(predictions == 1)}")
    if y_true is not None:
        accuracy = sum(predictions == y_true) / len(y_true)
        print(f"True enhancements: {sum(y_true == 1)}")
        print(f"Prediction accuracy: {sum(predictions == y_true)}/{len(y_true)} ({accuracy:.1%})")
    
    # Identify high-probability enhancement periods
    peak_times = storm_data.index[enhancement_prob > 0.5]
    if len(peak_times) > 0:
        print(f"\nHigh-probability enhancement periods:")
        for t in peak_times[:5]:  # Display first 5
            prob_value = enhancement_prob[storm_data.index == t][0]
            print(f"  {t}: {prob_value:.1%} probability")


def find_available_storms():
    """
    Identify storm events available in the dataset.
    
    Returns:
        list: Dates of storm events found in the dataset
    """
    df = pd.read_csv('flux_enhancement_labeled_fixed.csv', 
                     index_col=0, parse_dates=True)
    
    # Find periods where SYM/H < -50 (storm conditions)
    storm_mask = df['omni_SYM_H'] < -50
    storm_times = df.index[storm_mask]
    
    # Group consecutive hours into storm events
    storm_dates = []
    if len(storm_times) > 0:
        current_storm_start = storm_times[0]
        
        for i in range(1, len(storm_times)):
            # If gap > 2 hours, consider it a new storm
            if (storm_times[i] - storm_times[i-1]).total_seconds() > 7200:
                storm_dates.append(current_storm_start.date())
                current_storm_start = storm_times[i]
        
        storm_dates.append(current_storm_start.date())
    
    # Remove duplicates and sort
    storm_dates = sorted(list(set(storm_dates)))
    
    print(f"\nIdentified {len(storm_dates)} storm events in dataset:")
    for i, date in enumerate(storm_dates[:10]):  # Display first 10
        min_symh = df[df.index.date == date]['omni_SYM_H'].min()
        print(f"  {date}: Minimum SYM/H = {min_symh:.1f} nT")
    
    return storm_dates


def test_real_time_prediction():
    """
    Simulate real-time prediction using synthetic data.
    Demonstrates how the model would perform in operational conditions.
    """
    model, scaler = load_trained_model()
    
    # Create synthetic test data (simulating real satellite measurements)
    new_data = pd.DataFrame({
        'flux_filtered': [0.5],
        'lstar_filtered': [4.5],
        'MLT': [12.0],
        'alpha_local': [90.0],
        'omni_SYM_H': [-85],  # Storm condition
        'omni_B': [15.0],
        'omni_V': [450.0],
        'omni_n': [10.0],
        'hour': [14],
        'day_of_year': [100],
        'flux_rolling_mean_24h': [0.3],
        'flux_rolling_std_24h': [0.1],
        'flux_rolling_max_24h': [0.6],
        'symh_rolling_mean_24h': [-60],
        'symh_rolling_min_24h': [-85],
        'flux_lag_6h': [0.4],
        'flux_lag_12h': [0.35],
        'flux_lag_24h': [0.3],
        'symh_lag_6h': [-70],
        'symh_lag_12h': [-50],
        'flux_change_6h': [0.1],
        'flux_change_12h': [0.15],
        'hours_since_storm': [6]  # 6 hours after storm start
    })
    
    # Prepare data for prediction
    X_scaled = scaler.transform(new_data.values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Generate prediction
    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
    
    print("\nReal-Time Prediction Simulation:")
    print(f"Current conditions: SYM/H = {new_data['omni_SYM_H'].values[0]} nT (STORM CONDITION)")
    print(f"Hours since storm onset: {new_data['hours_since_storm'].values[0]}")
    print(f"Current particle flux: {new_data['flux_filtered'].values[0]}")
    print(f"\nModel Prediction: {'ENHANCEMENT DETECTED' if prediction == 1 else 'NORMAL CONDITIONS'}")
    print(f"Prediction confidence: {prob[0, prediction].item():.1%}")
    print(f"Enhancement probability: {prob[0, 1].item():.1%}")


def main():
    """
    Execute comprehensive model testing suite.
    """
    print("Flux Enhancement Model Testing Suite")
    print("=" * 50)
    
    # Identify available storm events
    storm_dates = find_available_storms()
    
    if len(storm_dates) > 0:
        # Test on the first available storm
        print(f"\nTesting model on storm event: {storm_dates[0]}")
        test_on_specific_storm(str(storm_dates[0]))
    
    # Demonstrate real-time prediction
    test_real_time_prediction()


if __name__ == "__main__":
    main()