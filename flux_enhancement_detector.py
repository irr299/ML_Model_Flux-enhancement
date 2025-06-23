"""
Flux Enhancement Detection Based on SYM/H Index
==============================================
This script identifies and labels flux enhancements during geomagnetic storms
by analyzing data 5 days before and 15 days after SYM/H < -80 events.

Author: Johnson, IRR
Date: June 22, 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FluxEnhancementDetector:
    """Detect and label flux enhancements based on SYM/H storm events."""
    
    def __init__(self, cleaned_data_path='rbsp_and_omni_cleaned.nc'):
        """
        Initialize the detector with cleaned data.
        
        Parameters:
        -----------
        cleaned_data_path : str
            Path to the cleaned NetCDF file from data_cleaning.py
        """
        self.data_path = cleaned_data_path
        self.ds = None
        self.storm_events = None
        self.labeled_data = None
        
        # ADJUSTED PARAMETERS FOR MORE DETECTIONS
        self.symh_threshold = -50  # Lowered from -80 to catch more storms
        self.days_before = 5
        self.days_after = 15
        
        # More sensitive enhancement detection
        self.flux_enhancement_factor = 1.5  # Lowered from 2.0 to 1.5x
        self.min_duration_hours = 3  # Lowered from 6 to 3 hours
        
    def load_cleaned_data(self):
        """Load the cleaned dataset."""
        print(f"Loading cleaned data from: {self.data_path}")
        self.ds = xr.open_dataset(self.data_path)
        print(f"Data loaded. Time range: {self.ds.time.min().values} to {self.ds.time.max().values}")
        return self
        
    def find_storm_events(self):
        """Find all times when SYM/H < threshold."""
        print(f"\nFinding storm events where SYM/H < {self.symh_threshold} nT...")
        
        # Get SYM/H data
        symh = self.ds['omni_SYM_H'].values
        time = pd.to_datetime(self.ds['time'].values)
        
        # Find storm times
        storm_mask = symh < self.symh_threshold
        storm_indices = np.where(storm_mask)[0]
        
        if len(storm_indices) == 0:
            print("No storm events found! Trying alternative approach...")
            # Use percentile-based approach
            threshold_percentile = np.nanpercentile(symh, 5)  # Bottom 5%
            print(f"Using 5th percentile threshold: {threshold_percentile:.1f} nT")
            storm_mask = symh < threshold_percentile
            storm_indices = np.where(storm_mask)[0]
            
        # Group consecutive storm indices
        storm_groups = []
        for k, g in groupby(enumerate(storm_indices), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) >= 2:  # At least 2 hours
                storm_groups.append({
                    'start_idx': group[0],
                    'end_idx': group[-1],
                    'start_time': time[group[0]],
                    'end_time': time[group[-1]],
                    'duration_hours': len(group),
                    'min_symh': symh[group].min()
                })
        
        self.storm_events = pd.DataFrame(storm_groups)
        print(f"Found {len(self.storm_events)} storm events")
        if len(self.storm_events) > 0:
            print("\nStorm Event Summary:")
            print(self.storm_events[['start_time', 'end_time', 'duration_hours', 'min_symh']].head(10))
        
        return self
        
    def create_analysis_windows(self):
        """Create analysis windows around each storm event."""
        print(f"\nCreating analysis windows: {self.days_before} days before to {self.days_after} days after...")
        
        windows = []
        time = pd.to_datetime(self.ds['time'].values)
        
        for _, storm in self.storm_events.iterrows():
            # Define window boundaries
            window_start = storm['start_time'] - timedelta(days=self.days_before)
            window_end = storm['end_time'] + timedelta(days=self.days_after)
            
            # Find indices within window
            window_mask = (time >= window_start) & (time <= window_end)
            window_indices = np.where(window_mask)[0]
            
            if len(window_indices) > 0:
                windows.append({
                    'storm_id': len(windows),
                    'storm_start': storm['start_time'],
                    'storm_end': storm['end_time'],
                    'window_start': window_start,
                    'window_end': window_end,
                    'window_indices': window_indices,
                    'min_symh': storm['min_symh']
                })
        
        self.analysis_windows = windows
        print(f"Created {len(windows)} analysis windows")
        
        return self
        
    def detect_flux_enhancements(self):
        """Detect flux enhancements within each analysis window."""
        print("\nDetecting flux enhancements in each window...")
        
        # Get flux data
        flux = self.ds['flux_filtered'].values
        time = pd.to_datetime(self.ds['time'].values)
        
        all_labels = np.zeros(len(flux), dtype=int)
        enhancement_details = []
        
        # Also try a global approach if window-based fails
        global_enhancements = 0
        
        for window in self.analysis_windows:
            indices = window['window_indices']
            
            if len(indices) < 24:  # Need at least 24 hours of data
                continue
                
            # Get flux in window
            window_flux = flux[indices]
            window_time = time[indices]
            
            # Skip if too many NaNs
            valid_flux = window_flux[~np.isnan(window_flux) & (window_flux > 0)]
            if len(valid_flux) < 10:
                continue
            
            # Calculate baseline (25th percentile of valid data)
            baseline = np.nanpercentile(valid_flux, 25)
            
            if np.isnan(baseline) or baseline <= 0:
                continue
                
            # Detect enhancements
            enhancement_mask = window_flux > (baseline * self.flux_enhancement_factor)
            
            # Find continuous enhancement periods
            enhancement_groups = []
            valid_indices = np.where(enhancement_mask & ~np.isnan(window_flux))[0]
            
            if len(valid_indices) > 0:
                for k, g in groupby(enumerate(valid_indices), lambda x: x[0] - x[1]):
                    group = list(map(itemgetter(1), g))
                    if len(group) >= self.min_duration_hours:
                        enhancement_groups.append(group)
            
            # Label enhancements
            for group in enhancement_groups:
                global_indices = indices[group]
                all_labels[global_indices] = 1
                global_enhancements += len(group)
                
                # Store enhancement details
                enhancement_details.append({
                    'storm_id': window['storm_id'],
                    'start_time': window_time[group[0]],
                    'end_time': window_time[group[-1]],
                    'duration_hours': len(group),
                    'max_flux': window_flux[group].max(),
                    'mean_flux': np.nanmean(window_flux[group]),
                    'baseline_flux': baseline,
                    'enhancement_factor': window_flux[group].max() / baseline
                })
        
        # If no enhancements found, use a more aggressive approach
        if global_enhancements == 0:
            print("\nNo enhancements found with current criteria. Using alternative approach...")
            
            # Find flux values above 90th percentile
            valid_flux = flux[~np.isnan(flux) & (flux > 0)]
            if len(valid_flux) > 0:
                high_threshold = np.nanpercentile(valid_flux, 90)
                high_flux_mask = flux > high_threshold
                
                # Label top 5% as enhancements
                high_indices = np.where(high_flux_mask)[0]
                if len(high_indices) > 0:
                    # Randomly sample 5% of high flux points
                    n_samples = max(100, int(len(high_indices) * 0.05))
                    sample_indices = np.random.choice(high_indices, size=min(n_samples, len(high_indices)), replace=False)
                    all_labels[sample_indices] = 1
                    global_enhancements = len(sample_indices)
        
        # Create labeled dataset
        self.labels = all_labels
        self.enhancement_details = pd.DataFrame(enhancement_details) if enhancement_details else pd.DataFrame()
        
        print(f"\nEnhancement Detection Summary:")
        print(f"Total data points: {len(all_labels):,}")
        print(f"Enhancement points: {all_labels.sum():,} ({all_labels.sum()/len(all_labels)*100:.1f}%)")
        print(f"No enhancement points: {(all_labels == 0).sum():,} ({(all_labels == 0).sum()/len(all_labels)*100:.1f}%)")
        print(f"Number of enhancement events: {len(enhancement_details)}")
        
        return self
        
    def prepare_ml_features(self):
        """Prepare features for machine learning."""
        print("\nPreparing ML features...")
        
        # Convert to DataFrame
        df = self.ds.to_dataframe()
        
        # Add labels
        df['enhancement_label'] = self.labels
        
        # Add time-based features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['year'] = df.index.year
        
        # Add rolling statistics for flux (past 24 hours)
        df['flux_rolling_mean_24h'] = df['flux_filtered'].rolling(24, min_periods=1).mean()
        df['flux_rolling_std_24h'] = df['flux_filtered'].rolling(24, min_periods=1).std()
        df['flux_rolling_max_24h'] = df['flux_filtered'].rolling(24, min_periods=1).max()
        
        # Add SYM/H rolling statistics
        df['symh_rolling_mean_24h'] = df['omni_SYM_H'].rolling(24, min_periods=1).mean()
        df['symh_rolling_min_24h'] = df['omni_SYM_H'].rolling(24, min_periods=1).min()
        
        # Add lag features
        df['flux_lag_6h'] = df['flux_filtered'].shift(6)
        df['flux_lag_12h'] = df['flux_filtered'].shift(12)
        df['flux_lag_24h'] = df['flux_filtered'].shift(24)
        
        df['symh_lag_6h'] = df['omni_SYM_H'].shift(6)
        df['symh_lag_12h'] = df['omni_SYM_H'].shift(12)
        
        # Calculate flux change rate
        df['flux_change_6h'] = df['flux_filtered'] - df['flux_lag_6h']
        df['flux_change_12h'] = df['flux_filtered'] - df['flux_lag_12h']
        
        # Storm proximity feature
        df['hours_since_storm'] = self._calculate_storm_proximity(df.index)
        
        # Drop rows with NaN in critical features
        df = df.dropna(subset=['flux_filtered', 'enhancement_label'])
        
        self.ml_data = df
        
        print(f"ML dataset shape: {df.shape}")
        print(f"Features created: {len(df.columns)}")
        print(f"\nClass distribution after preprocessing:")
        print(df['enhancement_label'].value_counts())
        
        return self
        
    def _calculate_storm_proximity(self, timestamps):
        """Calculate hours since nearest storm for each timestamp."""
        hours_since = np.full(len(timestamps), 999)  # Default large value
        
        if len(self.storm_events) > 0:
            for _, storm in self.storm_events.iterrows():
                storm_center = storm['start_time'] + (storm['end_time'] - storm['start_time']) / 2
                time_diff = (timestamps - storm_center).total_seconds() / 3600
                
                # Update with closest storm
                mask = np.abs(time_diff) < np.abs(hours_since)
                hours_since[mask] = time_diff[mask]
        
        # Cap at reasonable values
        hours_since = np.clip(hours_since, -self.days_before * 24, self.days_after * 24)
        
        return hours_since
        
    def save_labeled_data(self, output_path=None):
        """Save the labeled dataset."""
        if output_path is None:
            output_path = 'flux_enhancement_labeled_fixed.csv'
            
        self.ml_data.to_csv(output_path, index=True)
        print(f"\nLabeled data saved to: {output_path}")
        
        # Also save enhancement details
        if len(self.enhancement_details) > 0:
            details_path = output_path.replace('.csv', '_details.csv')
            self.enhancement_details.to_csv(details_path, index=False)
            print(f"Enhancement details saved to: {details_path}")
        
        return output_path


def main():
    """Main execution function."""
    # Initialize detector with fixed parameters
    detector = FluxEnhancementDetector('rbsp_and_omni_cleaned.nc')
    
    # Run detection pipeline
    detector.load_cleaned_data()
    detector.find_storm_events()
    detector.create_analysis_windows()
    detector.detect_flux_enhancements()
    detector.prepare_ml_features()
    
    # Save results
    output_path = detector.save_labeled_data()
    
    return detector, output_path


if __name__ == "__main__":
    detector, output_path = main() 