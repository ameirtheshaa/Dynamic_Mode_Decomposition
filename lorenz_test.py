import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import pandas as pd
from scipy.integrate import solve_ivp
from dmd import DMD
from dmd_plotting import DMDPlotter

def lorenz_system(t, xyz, sigma=10, beta=8/3, rho=28):
    """
    Lorenz system of differential equations.
    
    Parameters:
    -----------
    t : float
        Time (unused, but required by solve_ivp)
    xyz : array_like
        Point in three-dimensional space
    sigma, beta, rho : float
        Parameters of the Lorenz system
        
    Returns:
    --------
    dxyz : ndarray
        Derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

def generate_lorenz_data(t_span=(0, 50), n_time_points=5000, n_spatial_points=100, initial_state=None, sigma=10, beta=8/3, rho=28):
    """
    Generate data from the Lorenz system with multiple spatial points.
    
    Parameters:
    -----------
    t_span : tuple
        Start and end times for integration
    n_time_points : int
        Number of time points to sample
    n_spatial_points : int
        Number of spatial points to generate
    initial_state : array_like or None
        Initial state vector [x0, y0, z0] for the base trajectory
    sigma, beta, rho : float
        Parameters of the Lorenz system
        
    Returns:
    --------
    data : ndarray
        Data matrix with shape (N, m) where N is the number of spatial points and m is the number of time points
    spatial_coords : ndarray
        Matrix containing the 3D coordinates of each spatial point, shape (N, 3)
    t : ndarray
        Time points
    """
    import numpy as np
    from scipy.integrate import solve_ivp
    
    def lorenz_system(t, xyz, sigma, beta, rho):
        """The Lorenz system of ODEs."""
        x, y, z = xyz
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]  # Default initial state
    
    # Generate evenly spaced time points
    t = np.linspace(t_span[0], t_span[1], n_time_points)
    
    # Create slightly perturbed initial conditions for each spatial point
    np.random.seed(42)  # For reproducibility
    perturbations = np.random.normal(0, 0.1, (n_spatial_points, 3))
    initial_conditions = np.tile(initial_state, (n_spatial_points, 1)) + perturbations
    
    # Initialize data matrix and spatial coordinates
    data = np.zeros((n_spatial_points, n_time_points))
    spatial_coords = np.zeros((n_spatial_points, 3))
    
    # Solve the system for each spatial point
    for i in range(n_spatial_points):
        ic = initial_conditions[i]
        spatial_coords[i] = ic  # Store the initial position as the spatial coordinate
        
        sol = solve_ivp(
            lambda t, xyz: lorenz_system(t, xyz, sigma, beta, rho),
            t_span,
            ic,
            t_eval=t,
            method='RK45',
            rtol=1e-6
        )
        
        # Store the x-component of the solution as the data for this spatial point
        # You could also use y or z component, or some combination
        data[i, :] = sol.y[0, :]  # Using x-component
    
    return data, spatial_coords, t

def plot_lorenz_attractors(data, spatial_coords, t, reconstructed, future_states, output_dir):
    """
    Create specialized visualizations for the Lorenz system attractors.
    
    Parameters:
    -----------
    data : ndarray
        Original data matrix with shape (N, m) where N is spatial points and m is time points
    spatial_coords : ndarray
        Matrix containing the 3D coordinates of each spatial point, shape (N, 3)
    t : ndarray
        Time points
    reconstructed : ndarray
        Reconstructed data matrix with shape (N, m)
    future_states : ndarray
        Predicted future states with shape (N, m_future)
    output_dir : str
        Directory to save output plots
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # For the Lorenz attractor, we need to extract the 3D trajectories
    # Assuming the first 3 spatial points correspond to x, y, z components of a trajectory
    # If different, this should be adjusted based on your data structure
    
    # Extract the 3D trajectory from the data
    x_original = data[0, :]  # First spatial point (x-component)
    y_original = data[1, :]  # Second spatial point (y-component)
    z_original = data[2, :]  # Third spatial point (z-component)
    
    # Extract 3D trajectory from reconstructed data
    x_recon = reconstructed[0, :].real
    y_recon = reconstructed[1, :].real
    z_recon = reconstructed[2, :].real
    
    # Extract 3D trajectory from future states
    x_future = future_states[0, :].real
    y_future = future_states[1, :].real
    z_future = future_states[2, :].real
    
    # Plot the original and reconstructed attractors for comparison
    fig = plt.figure(figsize=(15, 6))
    
    # Original attractor
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_original, y_original, z_original, 'b-', lw=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Lorenz Attractor')
    
    # Reconstructed attractor
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_recon, y_recon, z_recon, 'r-', lw=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('DMD Reconstructed Lorenz Attractor')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lorenz_reconstruction_comparison.png")
    plt.close(fig)
    
    # Plot the predicted attractor
    fig = plt.figure(figsize=(15, 6))
    
    # Original attractor
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_original, y_original, z_original, 'b-', lw=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Lorenz Attractor')
    
    # Predicted attractor (future states)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_future, y_future, z_future, 'r-', lw=0.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('DMD Predicted Lorenz Attractor')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lorenz_prediction_attractor.png")
    plt.close(fig)
    
    # Plot all three variables over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    dt = t[1] - t[0]  # Time step
    original_time = t
    future_time = np.arange(0, future_states.shape[1]) * dt
    
    variables = [(x_original, x_recon, x_future, 'X'),
                (y_original, y_recon, y_future, 'Y'),
                (z_original, z_recon, z_future, 'Z')]
    
    for i, (orig, recon, fut, label) in enumerate(variables):
        # Plot original data
        axes[i].plot(original_time, orig, 'b-', label='Original')
        
        # Plot reconstructed data
        axes[i].plot(original_time, recon, 'g--', label='Reconstructed')
        
        # Plot prediction
        axes[i].plot(future_time, fut, 'r-', label='DMD Prediction')
        
        # Vertical line at the end of training data
        axes[i].axvline(x=original_time[-1], color='k', linestyle='--')
        
        axes[i].set_ylabel(f'{label} Coordinate')
        axes[i].grid(True)
        axes[i].legend()
    
    axes[-1].set_xlabel('Time')
    plt.suptitle('Lorenz System: Original vs. DMD Prediction')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lorenz_variables_prediction.png")
    plt.close(fig)
        
    # Create a document explaining the Lorenz system
    lorenz_info = """
        # Lorenz System DMD Analysis
                
        The Lorenz system is a simplified mathematical model for atmospheric convection, introduced by Edward Lorenz in 1963. It is described by the following system of ordinary differential equations:

        ```
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz
        ```

        Where:
        - σ (sigma) is the Prandtl number
        - ρ (rho) is the Rayleigh number
        - β (beta) is a physical proportion

        For the classical parameter values (σ=10, β=8/3, ρ=28), the system exhibits chaotic behavior, characterized by:
        - Sensitivity to initial conditions (the "butterfly effect")
        - Strange attractor geometry
        - Non-repeating, deterministic trajectories
        - Positive Lyapunov exponent

        ## DMD Analysis of Lorenz System

        Dynamic Mode Decomposition provides insight into:
        1. The dominant coherent structures (modes) in the Lorenz system
        2. Their oscillation frequencies and growth/decay rates
        3. How well a linear approximation can capture this nonlinear, chaotic system
        4. Prediction capabilities and limitations

        ## Physical Interpretation of DMD Modes in Lorenz System

        The DMD modes of the Lorenz system typically correspond to:
        - Mode pairs representing oscillatory behavior around the attractor wings
        - Modes capturing transitions between attractor lobes
        - Modes representing average flow or baseline state
        - Higher-order modes capturing finer details of the chaotic dynamics

        ## Lyapunov Exponents

        The Lorenz system is characterized by its Lyapunov exponents, which for the standard parameters (σ=10, β=8/3, ρ=28) are approximately:
        - λ₁ ≈ 0.906 (positive, indicating chaos)
        - λ₂ ≈ 0 (zero, indicating a conserved quantity)
        - λ₃ ≈ -14.572 (negative, indicating strong contraction)

        The presence of a positive Lyapunov exponent confirms the chaotic nature of the system and explains why long-term predictions are fundamentally limited even with advanced methods like DMD.
        """
    
    # Save the Lorenz-specific explanation
    with open(f"{output_dir}/lorenz_system_explanation.md", 'w') as f:
        f.write(lorenz_info)

def main():
    """
    Main function to demonstrate DMD analysis on the Lorenz system.
    """
    print("=" * 80)
    print("LORENZ SYSTEM - DYNAMIC MODE DECOMPOSITION (DMD) ANALYSIS")
    print("=" * 80)
    
    # Create output directory for saving results
    output_dir = "lorenz_dmd_results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")
    
    # Generate Lorenz system data
    print("\nGenerating Lorenz system data...")
    # Use slightly different parameter values to observe interesting dynamics
    lorenz_data, spatial_data, t = generate_lorenz_data(t_span=(0, 50), n_time_points=100, n_spatial_points=1000, 
                                         initial_state=[0.0, 0.0, 0.0],
                                         sigma=10, beta=8/3, rho=28)
    
    dt = t[1] - t[0]  # Time step between snapshots
    
    print(f"Data shape: {lorenz_data.shape}")
    print(f"Time range: [{t[0]}, {t[-1]}], dt={dt:.6f}")
    
    # Plot the Lorenz attractor for reference
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(lorenz_data[0, :], lorenz_data[1, :], lorenz_data[2, :], 'b-', lw=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Lorenz Attractor')
    plt.savefig(f"{output_dir}/lorenz_attractor.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 1: Initialize DMD model and DMDPlotter with different ranks
    # =========================================================================
    print("\nInitializing DMD model...")
    
    # Try different rank values for comparison
    ranks = [None, 2, 4, 6, 8, 10, 15]
    dmd_models = {}
    
    for rank in ranks:
        # Initialize DMD with data and given rank
        dmd_models[rank] = DMD(lorenz_data, rank=rank, dt=dt)
        print(f"Initialized DMD model with rank = {rank}, effective rank = {dmd_models[rank].effective_rank}")
    
    # Select the DMD model with optimal rank for the main analysis
    # For simplicity, we'll use the model with rank=None (automatic rank selection)
    dmd = dmd_models[None]
    
    # Initialize the plotter with the selected model
    plotter = DMDPlotter(dmd)
    
    # =========================================================================
    # Step 2: Analyze SVD and rank selection
    # =========================================================================
    print("\nAnalyzing SVD and rank selection...")
    
    # Plot singular values
    fig = plotter.plot_svd_analysis(figsize=(14, 6), save_path=f"{output_dir}/svd_analysis.png")
    plt.close(fig)
    
    # Compare reconstruction errors for different ranks
    errors = {}
    for rank, model in dmd_models.items():
        if rank is not None:  # Skip None rank
            recon = model.reconstruct()
            if np.isreal(lorenz_data).all():
                recon = recon.real
            error = np.linalg.norm(lorenz_data - recon) / np.linalg.norm(lorenz_data)
            errors[rank] = error
    
    # Plot reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.plot(list(errors.keys()), list(errors.values()), 'o-', markersize=8)
    plt.xlabel('Rank')
    plt.ylabel('Relative Reconstruction Error')
    plt.title('Reconstruction Error vs. Rank for Lorenz System')
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f"{output_dir}/rank_errors.png")
    plt.close()
    
    # Compute optimal rank for prediction
    print("\nComputing optimal rank for prediction...")
    optimal_rank, rank_errors = dmd.compute_optimal_prediction_rank(test_ratio=0.2, max_rank=15)
    print(f"Optimal rank for prediction: {optimal_rank}")
    
    # Plot prediction errors
    plt.figure(figsize=(10, 6))
    plt.plot(list(rank_errors.keys()), list(rank_errors.values()), 'o-', markersize=8)
    plt.xlabel('Rank')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs. Rank for Lorenz System')
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
    fig = plotter.plot_mode_amplitudes(figsize=(12, 6), n_modes=10, save_path=f"{output_dir}/mode_amplitudes.png")
    plt.close(fig)
    
    # Plot mode frequencies
    fig = plotter.plot_mode_frequencies(figsize=(12, 6), n_modes=10, save_path=f"{output_dir}/mode_frequencies_plot.png")
    plt.close(fig)
    
    # Plot mode growth rates
    fig = plotter.plot_growth_rates(figsize=(12, 6), n_modes=10, save_path=f"{output_dir}/mode_growth_rates.png")
    plt.close(fig)
    
    # Plot mode contributions
    fig = plotter.plot_mode_contributions(n_modes=8, figsize=(12, 10), save_path=f"{output_dir}/mode_contributions.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 4: Analyze significant modes for the Lorenz system
    # =========================================================================
    print("\nAnalyzing significant modes...")
    
    # Get mode significance
    sig_df = dmd.mode_significance()
    print("\nMost significant modes:")
    print(sig_df.head(5))
    
    # Save mode significance to CSV
    sig_df.to_csv(f"{output_dir}/mode_significance.csv", index=False)
    
    # Plot spatial structure of top modes (using DMDPlotter as in original example)
    fig = plotter.spatial_mode_visualization(mode_indices=sig_df.head(4)['Mode'].astype(int).tolist(), 
                                           figsize=(24, 16), save_path=f"{output_dir}/spatial_modes.png")
    plt.close(fig)
    
    # Check if modes are physical
    print("\nPhysical mode check:")
    for mode_idx in sig_df.head(5)['Mode'].astype(int):
        is_physical = dmd.is_mode_physical(mode_idx)
        print(f"Mode {mode_idx}: {'Physical' if is_physical else 'Non-physical'}")
    
    # =========================================================================
    # Step 5: Reconstruction and prediction for Lorenz system
    # =========================================================================
    print("\nPerforming reconstruction and prediction...")
    
    # Reconstruct the data
    reconstructed = dmd.reconstruct()
    if np.isreal(lorenz_data).all():
        reconstructed = reconstructed.real
    
    # Calculate reconstruction error
    reconstruction_error = np.linalg.norm(lorenz_data - reconstructed) / np.linalg.norm(lorenz_data)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Plot reconstruction error over time
    fig = plotter.plot_reconstruction_error(figsize=(12, 6), save_path=f"{output_dir}/reconstruction_error.png")
    plt.close(fig)
    
    # We'll keep these custom plots for the Lorenz attractor visualization for later
    # For now, continue with the standard plotter functions as in the original example
    
    # Predict future states
    n_snapshots = lorenz_data.shape[1]
    future_times = np.linspace(0, (n_snapshots + 500) * dt, 3500)
    future_states, confidence = dmd.forecast(future_times)
    
    if np.isreal(lorenz_data).all():
        future_states = future_states.real
    
    # Plot prediction for a specific point (following original example structure)
    spatial_point = 0  # Use the x-coordinate of the Lorenz system
    
    fig = plotter.plot_forecast(spatial_point=spatial_point, 
                              future_times=future_times, 
                              future_states=future_states, 
                              confidence=confidence,
                              figsize=(32, 16), 
                              save_path=f"{output_dir}/forecast.png")
    plt.close(fig)
    
    # =========================================================================
    # Step 6: Time dynamics and mode evolution for Lorenz
    # =========================================================================
    print("\nAnalyzing time dynamics and mode evolution...")
    
    # Plot time dynamics of significant modes
    fig = plotter.plot_time_dynamics(mode_indices=sig_df.head(4)['Mode'].astype(int).tolist(), 
                                   t_span=(0, (n_snapshots + 200) * dt),
                                   figsize=(14, 10), 
                                   save_path=f"{output_dir}/time_dynamics.png")
    plt.close(fig)
    
    # Plot eigenfunction evolution for a significant mode
    fig = plotter.plot_eigenfunction_evolution(eigenfunction_idx=sig_df.iloc[0]['Mode'].astype(int),
                                             t_span=(0, (n_snapshots + 200) * dt),
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

    plot_lorenz_attractors(lorenz_data, spatial_data, t, reconstructed, future_states, output_dir)
    
    # =========================================================================
    # Step 8: Generate comprehensive report for Lorenz system
    # =========================================================================
    print("\nGenerating comprehensive DMD report for Lorenz system...")
    
    # Generate diagnostic report
    report = dmd.generate_diagnostic_report()
    
    # Save report to file
    with open(f"{output_dir}/dmd_report.md", 'w') as f:
        f.write(report)
    
    # Generate mode explanations with Lorenz context
    explanations = dmd.explain_dmd_modes(n_modes=5)
    
    # Save explanations to file
    with open(f"{output_dir}/lorenz_mode_explanations.md", 'w') as f:
        f.write(explanations)
    
    # =========================================================================
    # Step 9: Summary and recommendations
    # =========================================================================
    print("\n" + "=" * 80)
    print("DMD ANALYSIS SUMMARY FOR LORENZ SYSTEM")
    print("=" * 80)
    
    print(f"\nAnalyzed Lorenz system with {lorenz_data.shape[0]} variables and {lorenz_data.shape[1]} time snapshots.")
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
    print("\nLorenz System Dynamics Assessment:")
    if spectral_radius > 1.0:
        print("- System exhibits UNSTABLE behavior (spectral radius > 1)")
        print("- This aligns with the chaotic nature of the Lorenz system")
        print("- Short-term predictions may be possible, but long-term forecasting is limited")
    else:
        print("- System appears STABLE in the DMD approximation (spectral radius <= 1)")
        print("- This suggests DMD is capturing an averaged, linearized behavior")
        print("- The actual Lorenz system is known to be chaotic and sensitive to initial conditions")
    
    print("\nConclusions for the Lorenz system DMD analysis:")
    print("- DMD has captured the dominant dynamics of the chaotic Lorenz system")
    print("- The chaotic nature of the system affects prediction accuracy beyond short timescales")
    print("- The most significant modes represent the dominant oscillatory patterns in the attractor")
    
    print("\nAll results saved to:", output_dir)
    print("\n" + "=" * 80)
    
    return dmd

if __name__ == "__main__":
    main()