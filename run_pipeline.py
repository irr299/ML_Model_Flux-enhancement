"""
Complete Pipeline for Flux Enhancement Detection
===============================================
This script orchestrates the entire machine learning pipeline from data
cleaning through model training and evaluation for flux enhancement detection.

Author: [Your Name]
Date: [Current Date]
"""

import subprocess
import sys
import os


def run_step(step_name, script_name, description):
    """
    Execute a pipeline step with comprehensive logging.
    
    Parameters:
        step_name (str): Human-readable name of the step
        script_name (str): Python script to execute
        description (str): Detailed description of the step
    
    Returns:
        bool: True if step completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Executing: {script_name}")
    print("-"*60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"\nSUCCESS: {step_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {step_name} failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    """
    Execute the complete flux enhancement detection pipeline.
    """
    print("""
    ================================================================
              FLUX ENHANCEMENT DETECTION ML PIPELINE
    ================================================================
    
    This pipeline implements a complete machine learning solution for
    detecting particle flux enhancements during geomagnetic storms.
    
    Pipeline Overview:
    - Data preprocessing and feature engineering
    - Storm event detection and labeling
    - Neural network model training
    - Performance evaluation and testing
    ================================================================
    """)
    
    # Verify data availability
    if not os.path.exists('rbsp_and_omni.nc'):
        print("\nWARNING: Raw data file 'rbsp_and_omni.nc' not found!")
        print("Please ensure the NetCDF file is in the current directory.")
        return
    
    # Step 1: Data Cleaning and Preprocessing
    success = run_step(
        "1. DATA CLEANING AND PREPROCESSING",
        "data_cleaning.py",
        """
        This step performs comprehensive data preprocessing:
        - Loads raw NetCDF satellite data (7+ years of measurements)
        - Selects 7.7 MeV energy channel at 90-degree pitch angle
        - Applies noise threshold filtering (0.063)
        - Implements L-shell filtering (L* >= 3)
        - Creates machine learning-ready features
        - Saves cleaned dataset as 'rbsp_and_omni_cleaned.nc'
        """
    )
    
    if not success:
        print("\nCRITICAL ERROR: Pipeline terminated due to data cleaning failure.")
        return
    
    # Step 2: Enhancement Detection and Labeling
    success = run_step(
        "2. ENHANCEMENT DETECTION AND LABELING",
        "flux_enhancement_detector.py",
        """
        This step implements the enhancement detection algorithm:
        - Identifies geomagnetic storms (SYM/H < -80 nT)
        - Creates analysis windows (5 days before to 15 days after storms)
        - Detects flux enhancements using statistical thresholds
        - Labels data: 1 = enhancement detected, 0 = no enhancement
        - Generates time-based and statistical features
        - Saves labeled dataset as 'flux_enhancement_labeled.csv'
        """
    )
    
    if not success:
        print("\nCRITICAL ERROR: Pipeline terminated due to enhancement detection failure.")
        return
    
    # Step 3: Neural Network Training
    success = run_step(
        "3. NEURAL NETWORK MODEL TRAINING",
        "flux_enhancement_nn.py",
        """
        This step trains the machine learning model:
        - Loads labeled dataset with engineered features
        - Splits data: 70% training, 15% validation, 15% testing
        - Trains feed-forward neural network (3 hidden layers)
        - Implements class imbalance handling with weighted loss
        - Performs comprehensive model evaluation
        - Saves trained model as 'flux_enhancement_model.pth'
        """
    )
    
    if not success:
        print("\nCRITICAL ERROR: Pipeline terminated due to model training failure.")
        return
    
    print("""
    ================================================================
                        PIPELINE COMPLETED SUCCESSFULLY
    ================================================================
    
    Output Files Generated:
    - rbsp_and_omni_cleaned.nc         (preprocessed data)
    - rbsp_and_omni_ml_ready.csv       (feature-engineered data)
    - flux_enhancement_labeled.csv      (labeled dataset)
    - flux_enhancement_labeled_details.csv (event analysis)
    - flux_enhancement_model.pth        (trained neural network)
    - flux_enhancement_scaler.pkl       (feature normalization)
    - training_history.png              (training performance plots)
    ================================================================
    """)
    
    print("\nMACHINE LEARNING METHODOLOGY:")
    print("""
    1. STORM EVENT IDENTIFICATION:
       - Monitors SYM/H index for values below -80 nT
       - These indicate severe geomagnetic disturbances
       - Triggers analysis windows around storm events
    
    2. TEMPORAL ANALYSIS WINDOWS:
       - Examines 20-day periods centered on storm events
       - Compares flux levels to pre-storm baseline conditions
       - Identifies statistically significant flux increases
    
    3. FEATURE ENGINEERING:
       - Rolling statistics capture temporal patterns
       - Lag features model flux evolution over time
       - Storm proximity features indicate timing relationships
       - Solar wind parameters provide context
    
    4. NEURAL NETWORK ARCHITECTURE:
       - Feed-forward network with 3 hidden layers
       - Dropout regularization prevents overfitting
       - Weighted loss function handles class imbalance
       - Binary classification: enhancement vs. normal conditions
    
    5. MODEL VALIDATION:
       - Cross-validation ensures robust performance
       - Confusion matrix analysis provides detailed metrics
       - ROC curves evaluate classification performance
       - Feature importance analysis reveals key drivers
    """)
    
    print("\nSCIENTIFIC APPLICATIONS:")
    print("""
    - Satellite Operations: Predict radiation hazards for spacecraft
    - Astronaut Safety: Early warning for space missions
    - Power Grid Protection: Monitor geomagnetic storm impacts
    - Aviation Safety: Alert for high-altitude flight operations
    - Research Applications: Automated space weather analysis
    """)


if __name__ == "__main__":
    main() 