"""
Satellite Flux Data Cleaning Pipeline
=====================================
This script cleans satellite flux data for machine learning applications.
Specifically designed for 7.7 MeV energy channel at 90-degree pitch angle.

Author: Johnson, IRR
Date: June 22, 2025
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path


class FluxDataCleaner:
    """Clean and prepare satellite flux data for ML analysis."""
    
    def __init__(self, file_path):
        """
        Initialize the data cleaner.
        
        Parameters:
        -----------
        file_path : str
            Path to the NetCDF file containing satellite data
        """
        self.file_path = file_path
        self.ds = None
        self.clean_data = None
        
        # Cleaning parameters
        self.target_energy = 7.7  # MeV
        self.noise_threshold = 0.063
        self.l_minimum = 3
        self.pitch_angle_index = 8  # 90 degrees
        
    def load_data(self):
        """Load the NetCDF dataset."""
        print(f"Loading data from: {self.file_path}")
        self.ds = xr.open_dataset(self.file_path)
        self.ds = self.ds.load()  # Load into memory
        print(f"Data loaded successfully. Shape: {self.ds.dims}")
        return self
        
    def find_energy_channel(self):
        """Find the closest energy channel to target energy."""
        energy_values = self.ds['E_values'].values
        self.energy_index = np.argmin(np.abs(energy_values - self.target_energy))
        self.actual_energy = energy_values[self.energy_index]
        
        print(f"Target energy: {self.target_energy} MeV")
        print(f"Closest energy channel: E={self.energy_index}, Energy={self.actual_energy} MeV")
        return self.energy_index, self.actual_energy
        
    def clean_flux_data(self):
        """Apply all cleaning steps to the flux data."""
        # Step 1: Select specific energy and pitch angle
        flux_selected = self.ds['Flux'].sel(E=self.energy_index, PA=self.pitch_angle_index)
        lstar_selected = self.ds['Lstar'].sel(PA=self.pitch_angle_index)
        
        # Step 2: Apply noise threshold (label noise as 0)
        flux_filtered = flux_selected.where(flux_selected >= self.noise_threshold, other=0)
        
        # Step 3: Apply L-shell filter
        lstar_mask = lstar_selected >= self.l_minimum
        flux_filtered_final = flux_filtered.where(lstar_mask, other=np.nan)
        lstar_filtered = lstar_selected.where(lstar_mask, other=np.nan)
        
        # Create clean dataset
        self.clean_data = xr.Dataset({
            'flux_filtered': flux_filtered_final,
            'lstar_filtered': lstar_filtered,
            'time': self.ds['time'],
            'alpha_local': self.ds['alpha_local'].sel(PA=self.pitch_angle_index),
            'MLT': self.ds['MLT'],
            'omni_SYM_H': self.ds['omni_SYM/H'],
            'omni_B': self.ds['omni_B'],
            'omni_V': self.ds['omni_V'],
            'omni_n': self.ds['omni_n']
        })
        
        self._print_cleaning_summary(flux_selected, flux_filtered_final)
        return self
        
    def _print_cleaning_summary(self, original_flux, cleaned_flux):
        """Print summary statistics of the cleaning process."""
        print("\n=== CLEANING SUMMARY ===")
        print(f"Energy channel: {self.actual_energy} MeV")
        print(f"Pitch angle: 90 degrees (PA index = {self.pitch_angle_index})")
        print(f"Noise threshold: {self.noise_threshold}")
        print(f"L-shell minimum: {self.l_minimum}")
        
        # Calculate statistics
        total_points = cleaned_flux.size
        valid_flux = ((cleaned_flux > 0) & ~cleaned_flux.isnull()).sum().item()
        noise_points = (cleaned_flux == 0).sum().item()
        nan_points = cleaned_flux.isnull().sum().item()
        
        print(f"\nData breakdown:")
        print(f"Total data points: {total_points:,}")
        print(f"Valid flux values: {valid_flux:,} ({valid_flux/total_points*100:.1f}%)")
        print(f"Noise (labeled as 0): {noise_points:,} ({noise_points/total_points*100:.1f}%)")
        print(f"Filtered out (NaN): {nan_points:,} ({nan_points/total_points*100:.1f}%)")
        
    def prepare_ml_dataset(self):
        """
        Prepare data for machine learning.
        Returns a pandas DataFrame with features and labels.
        """
        if self.clean_data is None:
            raise ValueError("Data must be cleaned first. Run clean_flux_data().")
            
        # Convert to DataFrame
        df = self.clean_data.to_dataframe()
        
        # Create binary labels for classification
        # Label 1: Valid flux (above threshold and L* >= 3)
        # Label 0: Noise or filtered data
        df['label'] = ((df['flux_filtered'] > 0) & ~df['flux_filtered'].isna()).astype(int)
        
        # Create features for ML
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['year'] = df.index.year
        
        # Handle missing values in features
        feature_cols = ['lstar_filtered', 'MLT', 'omni_SYM_H', 'omni_B', 'omni_V', 'omni_n']
        for col in feature_cols:
            if col in df.columns:
                df[f'{col}_fillna'] = df[col].fillna(df[col].median())
        
        print(f"\nML Dataset prepared. Shape: {df.shape}")
        print(f"Valid samples: {df['label'].sum():,}")
        print(f"Noise/filtered samples: {(df['label'] == 0).sum():,}")
        
        return df
        
    def save_clean_data(self, output_path=None):
        """Save the cleaned dataset."""
        if output_path is None:
            output_path = Path(self.file_path).stem + '_cleaned.nc'
            
        self.clean_data.to_netcdf(output_path)
        print(f"\nCleaned data saved to: {output_path}")
        
    def save_ml_dataset(self, df, output_path=None):
        """Save the ML-ready dataset."""
        if output_path is None:
            output_path = Path(self.file_path).stem + '_ml_ready.csv'
            
        df.to_csv(output_path, index=True)
        print(f"ML dataset saved to: {output_path}")


def main():
    """Main execution function."""
    # Update this path to your actual file location
    file_path = '/Users/infantronald/M.Sc. Astrophysics/Machine Learning/main_code/rbsp_and_omni.nc'
    
    # Initialize cleaner
    cleaner = FluxDataCleaner(file_path)
    
    # Run cleaning pipeline
    cleaner.load_data()
    cleaner.find_energy_channel()
    cleaner.clean_flux_data()
    
    # Prepare ML dataset
    ml_df = cleaner.prepare_ml_dataset()
    
    # Save cleaned data
    cleaner.save_clean_data()
    cleaner.save_ml_dataset(ml_df)
    
    return cleaner, ml_df


if __name__ == "__main__":
    cleaner, ml_df = main() 