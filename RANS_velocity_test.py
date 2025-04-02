import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
from dmd_plotting import DMDPlotter
from dmd import DMD

def main(data, dt=1, use_full_rank=True):
    """
    Main function to demonstrate the capabilities of the DMD class.
    
    This function:
    1. Sets up the analysis environment
    2. Initializes a DMD model with the provided data
    3. Performs comprehensive analysis using the DMDPlotter
    4. Generates visualizations and reports
    5. Demonstrates forecasting capabilities
    
    Parameters:
    -----------
    data : numpy.ndarray
        The data matrix with shape (n_features, n_snapshots)
    dt : float
        Time step between snapshots (default=1)
        
    Returns:
    --------
    dmd : DMD
        The trained DMD model
    """
    print("=" * 80)
    print("DYNAMIC MODE DECOMPOSITION (DMD) ANALYSIS")
    print("=" * 80)
    
    # Create output directory for saving results
    output_dir = "RANS_speed_dmd_results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")
    
    print(f"Data shape: {data.shape}")

    n_snapshots = data.shape[1]
    n_features = data.shape[0]
    
    # =========================================================================
    # Step 1: Initialize DMD model and DMDPlotter
    # =========================================================================
    print("\nInitializing DMD model...")
    
    # Use full rank as requested
    if use_full_rank:
        # Initialize DMD with full rank
        rank = None  # None will use the full rank
        dmd = DMD(data, rank=rank, dt=dt)
        print(f"Initialized DMD model with full rank, effective rank = {dmd.effective_rank}")
    else:
        # Try different rank values for comparison (original approach)
        ranks = [None, 5, 10, 15, 20]
        dmd_models = {}
        
        for rank in ranks:
            # Initialize DMD with data and given rank
            dmd_models[rank] = DMD(data, rank=rank, dt=dt)
            print(f"Initialized DMD model with rank = {rank}, effective rank = {dmd_models[rank].effective_rank}")
        
        # Select the DMD model with optimal rank for the main analysis
        dmd = dmd_models[None]
    
    # Initialize the plotter with the selected model
    plotter = DMDPlotter(dmd)
    
    # =========================================================================
    # Step 2: Analyze SVD
    # =========================================================================
    print("\nAnalyzing SVD...")
    
    # Plot singular values
    fig = plotter.plot_svd_analysis(figsize=(14, 6), save_path=f"{output_dir}/svd_analysis.png")
    plt.close(fig)
    
    if not use_full_rank:
        # Compare reconstruction errors for different ranks (only if not using full rank)
        errors = {}
        for rank, model in dmd_models.items():
            if rank is not None:  # Skip None rank
                recon = model.reconstruct()
                if np.isreal(data).all():
                    recon = recon.real
                error = np.linalg.norm(data - recon) / np.linalg.norm(data)
                errors[rank] = error
        
        # Plot reconstruction errors
        plt.figure(figsize=(10, 6))
        plt.plot(list(errors.keys()), list(errors.values()), 'o-', markersize=8)
        plt.xlabel('Rank')
        plt.ylabel('Relative Reconstruction Error')
        plt.title('Reconstruction Error vs. Rank')
        plt.grid(True)
        plt.yscale('log')
        plt.savefig(f"{output_dir}/rank_errors.png")
        plt.close()
        
        # Compute optimal rank for prediction
        print("\nComputing optimal rank for prediction...")
        optimal_rank, rank_errors = dmd.compute_optimal_prediction_rank(test_ratio=0.2, max_rank=20)
        print(f"Optimal rank for prediction: {optimal_rank}")
    else:
        # Skip rank comparison if using full rank
        optimal_rank = dmd.effective_rank
        print(f"Using full rank: {optimal_rank}")
    
    # Plot prediction errors (only if not using full rank)
    if not use_full_rank:
        plt.figure(figsize=(10, 6))
        plt.plot(list(rank_errors.keys()), list(rank_errors.values()), 'o-', markersize=8)
        plt.xlabel('Rank')
        plt.ylabel('Prediction Error')
        plt.title('Prediction Error vs. Rank')
        plt.grid(True)
        plt.yscale('log')
        plt.axvline(optimal_rank, color='r', linestyle='--', label=f'Optimal Rank: {optimal_rank}')
        plt.legend()
        plt.savefig(f"{output_dir}/optimal_rank.png")
        plt.close()
    
    # =========================================================================
    # Step 3: Analyze DMD spectrum and modes
    # =========================================================================
    print("\nAnalyzing DMD spectrum and modes...")
    
    # Plot DMD spectrum
    fig = plotter.plot_spectrum(figsize=(10, 8), save_path=f"{output_dir}/dmd_spectrum.png")
    plt.close(fig)
    
    # Plot continuous-time spectrum
    fig = plotter.plot_continuous_spectrum(figsize=(12, 8), save_path=f"{output_dir}/continuous_spectrum.png")
    plt.close(fig)
    
    # Plot 3D spectrum
    fig = plotter.plot_3d_spectrum(figsize=(12, 10), save_path=f"{output_dir}/3d_spectrum.png")
    plt.close(fig)
    
    # Plot complex alpha
    fig = plotter.plot_complex_alpha(figsize=(10, 8), annotate=True, save_path=f"{output_dir}/complex_alpha.png")
    plt.close(fig)
    
    # Analyze mode frequencies and amplitudes
    print("\nMode frequency analysis:")
    mode_freq_df = dmd.mode_frequencies()
    print(mode_freq_df.head(10))
    
    # Save mode information to CSV
    mode_freq_df.to_csv(f"{output_dir}/mode_frequencies.csv", index=False)
    
    # Plot mode amplitudes
    fig = plotter.plot_mode_amplitudes(figsize=(12, 6), n_modes=15, save_path=f"{output_dir}/mode_amplitudes.png")
    plt.close(fig)
    
    # Plot mode frequencies
    fig = plotter.plot_mode_frequencies(figsize=(12, 6), n_modes=15, save_path=f"{output_dir}/mode_frequencies_plot.png")
    plt.close(fig)
    
    # Plot mode growth rates
    fig = plotter.plot_growth_rates(figsize=(12, 6), n_modes=15, save_path=f"{output_dir}/mode_growth_rates.png")
    plt.close(fig)
    
    # Plot mode contributions
    fig = plotter.plot_mode_contributions(n_modes=10, figsize=(12, 10), save_path=f"{output_dir}/mode_contributions.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 4: Analyze significant modes
    # =========================================================================
    print("\nAnalyzing significant modes...")
    
    # Get mode significance
    sig_df = dmd.mode_significance()
    print("\nMost significant modes:")
    print(sig_df.head(5))
    
    # Save mode significance to CSV
    sig_df.to_csv(f"{output_dir}/mode_significance.csv", index=False)
    
    # Plot spatial structure of top modes
    fig = plotter.spatial_mode_visualization(mode_indices=sig_df.head(4)['Mode'].astype(int).tolist(), 
                                           figsize=(24, 16), save_path=f"{output_dir}/spatial_modes.png")
    plt.close(fig)
    
    # Check if modes are physical
    print("\nPhysical mode check:")
    for mode_idx in sig_df.head(5)['Mode'].astype(int):
        is_physical = dmd.is_mode_physical(mode_idx)
        print(f"Mode {mode_idx}: {'Physical' if is_physical else 'Non-physical'}")
    
    # =========================================================================
    # Step 5: Reconstruction and prediction
    # =========================================================================
    print("\nPerforming reconstruction and prediction...")
    
    # Reconstruct the data
    reconstructed = dmd.reconstruct()
    if np.isreal(data).all():
        reconstructed = reconstructed.real
    
    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(data - reconstructed) / np.linalg.norm(data)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Plot reconstruction error over time
    fig = plotter.plot_reconstruction_error(figsize=(12, 6), save_path=f"{output_dir}/reconstruction_error.png")
    plt.close(fig)
    
    # Predict future states
    future_times = np.linspace(0, (n_snapshots + 20) * dt, 100)
    future_states, confidence = dmd.forecast(future_times)
    
    if np.isreal(data).all():
        future_states = future_states.real
    
    # Plot prediction for a specific spatial point
    spatial_point = n_features // 2  # Middle point
    
    fig = plotter.plot_forecast(spatial_point=spatial_point, 
                              future_times=future_times, 
                              future_states=future_states, 
                              confidence=confidence,
                              figsize=(32,16), 
                              save_path=f"{output_dir}/forecast.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 6: Time dynamics and mode evolution
    # =========================================================================
    print("\nAnalyzing time dynamics and mode evolution...")
    
    # Plot time dynamics of significant modes
    fig = plotter.plot_time_dynamics(mode_indices=sig_df.head(4)['Mode'].astype(int).tolist(), 
                                   t_span=(0, (n_snapshots + 10) * dt),
                                   figsize=(14, 10), 
                                   save_path=f"{output_dir}/time_dynamics.png")
    plt.close(fig)
    
    # Plot eigenfunction evolution for a significant mode
    fig = plotter.plot_eigenfunction_evolution(eigenfunction_idx=sig_df.iloc[0]['Mode'].astype(int),
                                             t_span=(0, (n_snapshots + 10) * dt),
                                             figsize=(12, 6),
                                             save_path=f"{output_dir}/eigenfunction_evolution.png")
    plt.close(fig)
    
    # Create phase portrait for two significant modes
    top_modes = sig_df.head(2)['Mode'].astype(int).tolist()
    fig = plotter.plot_mode_phase_portrait(mode_idx1=top_modes[0], mode_idx2=top_modes[1], 
                                        figsize=(10, 8),
                                        save_path=f"{output_dir}/phase_portrait.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 7: Advanced DMD analysis
    # =========================================================================
    print("\nPerforming advanced DMD analysis...")
    
    # Check eigenvalues
    spectral_radius = dmd.eigenvalue_check()
    
    # Check mode orthogonality
    gram = dmd.modes_orthogonality()
    
    # Compute Koopman modes
    koopman_modes = dmd.compute_koopman_modes()
    print(f"Koopman modes shape: {koopman_modes.shape}")
    
    # Compute exact DMD modes
    exact_modes = dmd.compute_exact_dmd()
    print(f"Exact DMD modes shape: {exact_modes.shape}")
    
    # =========================================================================
    # Step 8: Generate comprehensive report
    # =========================================================================
    print("\nGenerating comprehensive DMD report...")
    
    # Generate diagnostic report
    report = dmd.generate_diagnostic_report()
    
    # Save report to file
    with open(f"{output_dir}/dmd_report.md", 'w') as f:
        f.write(report)
    
    # Generate mode explanations
    explanations = dmd.explain_dmd_modes(n_modes=5)
    
    # Save explanations to file
    with open(f"{output_dir}/mode_explanations.md", 'w') as f:
        f.write(explanations)
    
    # =========================================================================
    # Step 9: Summary and recommendations
    # =========================================================================
    print("\n" + "=" * 80)
    print("DMD ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nAnalyzed data with {n_features} spatial points and {n_snapshots} time snapshots.")
    print(f"Optimal DMD rank: {optimal_rank}")
    
    print("\nTop 3 significant modes:")
    for i, (_, row) in enumerate(sig_df.head(3).iterrows()):
        mode_idx = int(row['Mode'])
        freq = mode_freq_df.loc[mode_freq_df['Mode'] == mode_idx, 'Frequency'].values[0]
        growth = mode_freq_df.loc[mode_freq_df['Mode'] == mode_idx, 'Growth_Rate'].values[0]
        print(f"  {i+1}. Mode {mode_idx}: Significance = {row['Significance']:.4f}, "
              f"Frequency = {freq:.4f}, Growth Rate = {growth:.4f}")
    
    print(f"\nOverall reconstruction error: {reconstruction_error:.6f}")
    
    # System stability assessment
    if spectral_radius > 1.0:
        print("\nSystem is UNSTABLE (spectral radius > 1)")
        print("Dominant growing modes detected.")
    else:
        print("\nSystem is STABLE (spectral radius <= 1)")
        print("All modes are decaying or marginally stable.")
    
    print("\nAll results saved to:", output_dir)
    print("\n" + "=" * 80)
    
    return dmd


if __name__ == "__main__":
    # Load the data using the provided preprocessing steps
    print("Loading and preprocessing data...")
    
    # Load all data files
    data = np.load('data/data.npy')
    xyz = np.load('data/xyz.npy')
    val_data = np.load('data/val_data.npy')
    data_all = np.load('data/data_all.npy')
    
    # Normalize data
    data_mean = (np.min(data, axis=(0,1)) + np.max(data, axis=(0,1)))/2
    data_std = (np.max(data, axis=(0,1)) - np.min(data, axis=(0,1)))/2
    data_ = (data - data_mean) / data_std
    val_data_ = (val_data - data_mean) / data_std
    data_all_ = (data_all - data_mean) / data_std
    
    # Normalize spatial coordinates
    xyz_mean = (np.min(xyz, axis=0) + np.max(xyz, axis=0))/2
    xyz_std = (np.max(xyz, axis=0) - np.min(xyz, axis=0))/2
    xyz_ = (xyz - xyz_mean) / xyz_std
    
    # Print data information
    print('Data Shape:', data.shape)
    print('Validation Data Shape:', val_data.shape)
    print('Spatial Data Shape:', xyz.shape)
    print('Normalized Data Shape:', data_.shape)
    print('Normalized Validation Data Shape:', val_data_.shape)
    print('Normalized Spatial Data Shape:', xyz_.shape)
    print('Data Mean, Std Dev:', data_mean, data_std)
    print('XYZ Data Mean, Std Dev:', xyz_mean, xyz_std)
    print('Normalized Data Range:', np.min(data_), np.max(data_))
    
    # Extract velocity components from the data
    N, m, k = data.shape
    print('Data Size:', N, m, k)
    
    # Extract velocity components (using index 1, 2, 3 as mentioned in your preprocessing)
    data_u = data_[:, :, 1].reshape(-1, m)
    data_v = data_[:, :, 2].reshape(-1, m)
    data_w = data_[:, :, 3].reshape(-1, m)
    
    print('Velocity Components Shapes:', data_u.shape, data_v.shape, data_w.shape)
    
    # Calculate velocity magnitude
    data_v_mag = np.sqrt(data_u**2 + data_v**2 + data_w**2)
    print('Velocity Magnitude Shape:', data_v_mag.shape)
    
    # Choose which data to analyze with DMD
    # Options:
    # 1. Use velocity magnitude
    dmd_data = data_v_mag
    
    # 2. Use one velocity component
    # dmd_data = data_u
    
    # 3. Use specific output parameter
    # param_index = 1  # Choose the parameter index (0, 1, 2, 3, etc.)
    # dmd_data = data_[:, :, param_index].reshape(-1, m)
    
    # 4. Use all data (reshaped to 2D)
    # dmd_data = data_.reshape(-1, m)
    
    print("Data prepared for DMD analysis:", dmd_data.shape)
    
    # Define time step (angle)
    dt = 15
    
    # Run DMD analysis with full rank
    dmd = main(dmd_data, dt=dt, use_full_rank=True)
    
    # If you want to show the plots instead of just saving them
    # plt.show()