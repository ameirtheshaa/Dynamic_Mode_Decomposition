import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set better default styles for plotting with improved spacing
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [4, 3]  # More balanced default size
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12  # Reduced from 12 to prevent overlap
plt.rcParams['axes.labelsize'] = 14  # Reduced from 14
plt.rcParams['axes.titlesize'] = 16  # Reduced from 16
plt.rcParams['xtick.labelsize'] = 12  # Reduced from 12
plt.rcParams['ytick.labelsize'] = 12  # Reduced from 12
plt.rcParams['legend.fontsize'] = 12  # Reduced from 12

# Add padding around figure to prevent cutoff
plt.rcParams['figure.constrained_layout.use'] = False
plt.rcParams['figure.autolayout'] = False  # Don't use tight_layout automatically

# Add these crucial spacing parameters
plt.rcParams['figure.subplot.top'] = 0.88
plt.rcParams['figure.subplot.bottom'] = 0.12
plt.rcParams['figure.subplot.left'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.92
plt.rcParams['figure.subplot.wspace'] = 0.25  # Width space between subplots
plt.rcParams['figure.subplot.hspace'] = 0.35  # Height space between subplots


class DMDPlotter:
    """
    A class for visualizing Dynamic Mode Decomposition (DMD) results.
    
    This class provides methods for various DMD visualizations including:
    - Singular value analysis
    - DMD spectrum visualizations
    - Mode analysis plots
    - Reconstruction and forecast visualizations
    - Time dynamics and phase portraits
    """
    
    def __init__(self, dmd_model):
        """
        Initialize the DMDPlotter with a DMD model.
        
        Parameters:
        -----------
        dmd_model : DMD
            The DMD model object containing modes, eigenvalues, etc.
        """
        self.dmd = dmd_model
        
        # Set custom matplotlib params for this instance
        # These settings will ensure better spacing and prevent text overlapping
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = [16, 10]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['font.family'] = 'DejaVu Sans'  # This font supports more special characters
    
    def spatial_mode_visualization(self, mode_indices, figsize=(10, 8), save_path=None):
        """
        Visualize the spatial structure of selected DMD modes.
        
        This visualization reveals the spatial patterns captured by DMD:
        1. Real and imaginary parts show spatial structure of oscillatory modes
        2. Magnitude shows where the mode has strongest influence
        3. Phase information reveals propagation patterns and standing waves
        
        Parameters:
        -----------
        mode_indices : list
            List of mode indices to visualize.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        modes = self.dmd.phi
        eigenvalues = self.dmd.eigenvalues
        amplitudes = self.dmd.alpha
        n_modes = len(mode_indices)
        
        fig = plt.figure(figsize=figsize, dpi=450)
        
        # Create a GridSpec layout with enough rows for modes plus explanation
        # Increased row height for explanation
        gs = gridspec.GridSpec(n_modes + 1, 3, height_ratios=[3] * n_modes + [1.2])
        
        # For each mode, create visualizations of real, imaginary, and magnitude/phase
        for i, mode_idx in enumerate(mode_indices):
            mode = modes[:, mode_idx]
            eigenvalue = eigenvalues[mode_idx]
            amplitude = amplitudes[mode_idx]
            
            # Calculate frequency and growth rate
            lambda_c = np.log(eigenvalue) / self.dmd.dt
            frequency = np.abs(np.imag(lambda_c)) / (2 * np.pi)
            growth_rate = np.real(lambda_c)
            
            # Real part
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(np.real(mode), 'b-', linewidth=2)
            ax1.set_title(f'Mode {mode_idx} - Real Part', fontsize=11)  # Reduced from 12
            
            # Only add x label to bottom row
            if i == n_modes - 1:
                ax1.set_xlabel('Spatial Index')
            ax1.set_ylabel('Amplitude')
            ax1.grid(True, alpha=0.3)
            
            # Add eigenvalue info - more compact
            mode_info = (f"f={frequency:.3f}Hz, gr={growth_rate:.3f}\n"
                        f"|λ|={np.abs(eigenvalue):.3f}, |α|={np.abs(amplitude):.3f}")
            
            # Color-code based on stability
            if np.abs(eigenvalue) > 1.001:
                box_color = '#ffcccc'  # Light red for unstable
            elif np.abs(eigenvalue) < 0.999:
                box_color = '#ccffcc'  # Light green for stable
            else:
                box_color = '#ffffcc'  # Light yellow for marginal
                
            ax1.text(0.05, 0.95, mode_info, transform=ax1.transAxes,
                    va='top', ha='left', fontsize=10,  # Reduced from 10
                    bbox=dict(boxstyle="round,pad=0.2", fc=box_color, ec="gray", alpha=0.8))
            
            # Imaginary part
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.plot(np.imag(mode), 'r-', linewidth=2)
            ax2.set_title(f'Mode {mode_idx} - Imaginary Part', fontsize=11)  # Reduced from 12
            
            # Only add x label to bottom row
            if i == n_modes - 1:
                ax2.set_xlabel('Spatial Index')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)
            
            # Magnitude and phase
            ax3 = fig.add_subplot(gs[i, 2])
            
            # Create twin axes for magnitude and phase
            ax3b = ax3.twinx()
            
            # Plot magnitude
            mag_line, = ax3.plot(np.abs(mode), 'g-', linewidth=2, label='Magnitude')
            ax3.set_ylabel('Magnitude', color='g')
            ax3.tick_params(axis='y', labelcolor='g')
            
            # Plot phase
            phase_line, = ax3b.plot(np.angle(mode), 'mo--', linewidth=1.5, alpha=0.7, label='Phase')
            ax3b.set_ylabel('Phase (rad)', color='m')  # Shortened from 'radians'
            ax3b.tick_params(axis='y', labelcolor='m')
            
            # Add horizontal lines for ±π phase reference
            ax3b.axhline(y=np.pi, color='m', linestyle=':', alpha=0.5)
            ax3b.axhline(y=-np.pi, color='m', linestyle=':', alpha=0.5)
            ax3b.set_ylim(-np.pi-0.5, np.pi+0.5)
            
            ax3.set_title(f'Mode {mode_idx} - Magnitude & Phase', fontsize=11)  # Reduced from 12
            
            # Only add x label to bottom row
            if i == n_modes - 1:
                ax3.set_xlabel('Spatial Index')
            ax3.grid(True, alpha=0.3)
            
            # Add legend - smaller font
            lines = [mag_line, phase_line]
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper right', fontsize=8)  # Reduced fontsize
            
            # Identify propagation patterns based on phase gradient
            phase = np.unwrap(np.angle(mode))
            if len(phase) > 10:
                # Simple check for traveling waves - consistent phase gradient
                phase_diff = np.diff(phase)
                mean_phase_diff = np.mean(phase_diff)
                std_phase_diff = np.std(phase_diff)
                
                if std_phase_diff < 0.5 * np.abs(mean_phase_diff):
                    # Consistent phase gradient suggests traveling wave
                    if mean_phase_diff > 0:
                        pattern = "→ Rightward traveling wave"
                    else:
                        pattern = "← Leftward traveling wave"
                elif np.max(np.abs(phase)) < 0.2:  # Very small phase variation
                    pattern = "Standing wave (in phase)"
                elif np.any(np.abs(np.diff(np.sign(np.diff(phase)))) > 0):
                    pattern = "Complex wave pattern"
                else:
                    pattern = "Mixed wave behavior"
                    
                ax3.text(0.5, 0.03, pattern, transform=ax3.transAxes,
                       ha='center', va='bottom', fontsize=8,  # Reduced from 10
                       bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.8))
        
        # Add explanation at the bottom - more compact
        ax_exp = fig.add_subplot(gs[-1, :])
        explanation = """Spatial Mode Visualization:
            • Real Part (left): In-phase spatial pattern
            • Imaginary Part (middle): 90° phase-shifted pattern
            • Magnitude & Phase (right): Strength and timing patterns
            • Traveling waves: consistent phase gradients across space
            • Standing waves: regions with same phase separated by nodes
            • Oscillatory modes combine real & imaginary patterns as they evolve"""
        
        ax_exp.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=10,  # Reduced from 11
                   bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        ax_exp.axis('off')
        
        # # Increase spacing between rows and columns
        # fig.subplots_adjust(hspace=0.45, wspace=0.35)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=450, bbox_inches='tight')
            
        return fig
    
    def plot_time_dynamics(self, mode_indices, t_span=None, figsize=(14, 14), save_path=None):
        """
        Plot the time dynamics of selected DMD modes.
        
        This visualization shows how individual modes evolve over time:
        1. Real and imaginary components reveal oscillatory behavior
        2. Magnitude shows overall growth/decay trends
        3. Frequency and growth rate parameters help interpret physical meaning
        
        Parameters:
        -----------
        mode_indices : list
            List of mode indices to visualize.
        t_span : tuple, optional
            Time span (t_start, t_end) for visualization.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Time parameters
        dt = self.dmd.dt
        n_snapshots = self.dmd.data.shape[1]
        
        if t_span is None:
            t_span = (0, n_snapshots * dt)
        
        # Create time vector
        t = np.linspace(t_span[0], t_span[1], 200)  # Increased resolution for smoother plots
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        
        # Calculate how many rows and columns we need
        n_modes = len(mode_indices)
        n_cols = min(2, n_modes)
        n_rows = int(np.ceil(n_modes / n_cols))
        
        # Add a title to the figure
        fig.suptitle('DMD Mode Time Dynamics', fontsize=15, fontweight='bold', y=0.98)  # Reduced from 16
        
        # Create grid for subplots - more space for explanation
        gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[3] * n_rows + [1])
        
        # For each mode, create a subplot with enhanced visualization
        for i, mode_idx in enumerate(mode_indices):
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            # Get the eigenvalue for this mode
            eigenvalue = self.dmd.eigenvalues[mode_idx]
            
            # Calculate continuous-time eigenvalue
            lambda_c = np.log(eigenvalue) / dt
            
            # Calculate frequency and growth rate
            frequency = np.abs(np.imag(lambda_c)) / (2 * np.pi)
            growth_rate = np.real(lambda_c)
            
            # Calculate amplitude
            amplitude = np.abs(self.dmd.alpha[mode_idx])
            
            # Calculate normalized amplitude (for comparison across modes)
            norm_amplitude = amplitude / np.max(np.abs(self.dmd.alpha)) if np.max(np.abs(self.dmd.alpha)) > 0 else amplitude
            
            # Calculate time dynamics: alpha * exp(lambda * t)
            time_dynamics = amplitude * np.exp(lambda_c * t)
            
            # Plot time dynamics
            real_line, = ax.plot(t, np.real(time_dynamics), '-', linewidth=1.5, label='Real')  # Thinner line, shorter label
            imag_line, = ax.plot(t, np.imag(time_dynamics), '--', linewidth=1.5, label='Imag')  # Thinner line, shorter label
            mag_line, = ax.plot(t, np.abs(time_dynamics), ':', linewidth=1.5, label='Mag')  # Thinner line, shorter label
            
            # Add vertical line for training data end
            if n_snapshots * dt <= t_span[1]:
                ax.axvline(x=n_snapshots * dt, color='red', linestyle='-', linewidth=1, 
                         label='End of Training')  # Thinner line, shorter label
                
                # Add shaded regions for training and prediction
                ax.axvspan(0, n_snapshots * dt, alpha=0.1, color='blue')
                ax.axvspan(n_snapshots * dt, t_span[1], alpha=0.1, color='green')
            
            # Determine stability and add label
            if np.abs(eigenvalue) > 1.001:  # Adding small buffer for numerical precision
                stability = "UNSTABLE"
                stability_color = "red"
            elif np.abs(eigenvalue) < 0.999:
                stability = "STABLE"
                stability_color = "green"
            else:
                stability = "MARGINAL"
                stability_color = "orange"
            
            # Create a more compact title with key parameters
            title = (f"Mode {mode_idx} - {stability}\n"
                    f"f={frequency:.3f}Hz, gr={growth_rate:.3f}, "
                    f"a={norm_amplitude:.3f}")  # Shortened labels, fewer decimals
            
            ax.set_title(title, fontsize=10)  # Reduced from 11
            ax.text(0.5, 0.02, stability, transform=ax.transAxes, ha='center',
                  fontsize=10, fontweight='bold', color=stability_color)  # Reduced from 12
            
            # Add gridlines and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            
            # Add smaller tick labels
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the first subplot to avoid clutter
            if i == 0:
                ax.legend(loc='best', fontsize=8)  # Reduced fontsize
        
        # Add the explanation text at the bottom spanning all columns - more compact
        ax_exp = fig.add_subplot(gs[-1, :])
        explanation = """Time Dynamics Explained:
        • Blue (solid): Real part - in-phase component
        • Orange (dashed): Imaginary part - 90° phase-shifted component
        • Green (dotted): Magnitude - amplitude envelope (growth/decay trend)
        • Growing modes (|λ| > 1) increase in amplitude, decaying modes (|λ| < 1) decrease
        • Frequency (f) relates to eigenvalue position on unit circle
        • Each mode contributes based on its amplitude and spatial structure"""
        
        ax_exp.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=9,  # Reduced from 11
                   bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        ax_exp.axis('off')
        
        # Increase spacing between subplots
        fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_eigenfunction_evolution(self, eigenfunction_idx, t_span=None, figsize=(12, 8), save_path=None):
        """
        Plot the evolution of a specific DMD eigenfunction over time.
        
        Parameters:
        -----------
        eigenfunction_idx : int
            Index of the eigenfunction to visualize.
        t_span : tuple, optional
            Time span (t_start, t_end) for visualization.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Time parameters
        dt = self.dmd.dt
        n_snapshots = self.dmd.data.shape[1]
        
        if t_span is None:
            t_span = (0, n_snapshots * dt)
        
        # Create time vector
        t = np.linspace(t_span[0], t_span[1], 100)
        
        # Get the eigenvalue for this mode
        eigenvalue = self.dmd.eigenvalues[eigenfunction_idx]
        
        # Calculate continuous-time eigenvalue
        lambda_c = np.log(eigenvalue) / dt
        
        # Calculate time dynamics: exp(lambda * t)
        time_dynamics = np.exp(lambda_c * t)
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the real and imaginary parts over time
        ax.plot(t, np.real(time_dynamics), np.imag(time_dynamics), 'b-', linewidth=1.5)  # Reduced linewidth
        
        # Mark start and end points
        ax.scatter([t[0]], [np.real(time_dynamics[0])], [np.imag(time_dynamics[0])], 
                  c='g', s=80, label='Start')  # Reduced from 100
        ax.scatter([t[-1]], [np.real(time_dynamics[-1])], [np.imag(time_dynamics[-1])], 
                  c='r', s=80, label='End')  # Reduced from 100
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Real Part')
        ax.set_zlabel('Imaginary Part')
        ax.set_title(f'Eigenfunction {eigenfunction_idx} Evolution')
        ax.legend(fontsize=9)  # Added fontsize
        
        # Add better spacing
        plt.tight_layout(pad=1.0)
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_mode_phase_portrait(self, mode_idx1, mode_idx2, figsize=(14, 12), save_path=None):
        """
        Create a phase portrait for two DMD modes.
        
        This visualization shows the relationship between two modes:
        1. Trajectories reveal how modes interact and evolve together
        2. Closed orbits indicate periodic behavior
        3. Spirals indicate growth or decay
        4. Direction of motion shows temporal progression
        
        Parameters:
        -----------
        mode_idx1 : int
            Index of the first mode.
        mode_idx2 : int
            Index of the second mode.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Get original data and modes
        data = self.dmd.data
        modes = self.dmd.phi
        eigenvalues = self.dmd.eigenvalues
        
        # Project data onto the two modes
        projection1 = np.abs(np.dot(data.T, modes[:, mode_idx1]))
        projection2 = np.abs(np.dot(data.T, modes[:, mode_idx2]))
        
        # Get complex projections for phase information if needed
        complex_proj1 = np.dot(data.T, modes[:, mode_idx1])
        complex_proj2 = np.dot(data.T, modes[:, mode_idx2])
        
        # Initialize figure
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        
        # Create main phase portrait plot
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Create colormap for time evolution
        time_norm = Normalize(vmin=0, vmax=len(projection1)-1)
        cmap = plt.cm.viridis
        
        # Plot each point with color representing time - FIX for c/color warning
        for i in range(len(projection1)-1):
            # Convert normalized time to color using the colormap
            color_val = cmap(time_norm(i))
            
            # Use 'color' instead of 'c' parameter
            ax_main.scatter(projection1[i], projection2[i], 
                         color=color_val,  # Changed from c=cmap(time_norm(i))
                         s=60, edgecolor='k', linewidth=0.5)
            
            # Add arrow to show direction
            ax_main.arrow(projection1[i], projection2[i], 
                       projection1[i+1]-projection1[i], projection2[i+1]-projection2[i],
                       head_width=0.015*max(projection1), head_length=0.015*max(projection2),
                       fc='k', ec='k', alpha=0.5)
        
        # Plot the last point - FIX for c/color warning
        end_color = cmap(time_norm(len(projection1)-1))
        ax_main.scatter(projection1[-1], projection2[-1], 
                     color=end_color,  # Changed from c=cmap(time_norm(len(projection1)-1))
                     s=80, edgecolor='k', linewidth=1.5, label='End Point')
        
        # Add colorbar for time
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=time_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_main)
        cbar.set_label('Time Index', fontsize=10)
        
        # Calculate and display correlation coefficient
        corr_coef = np.corrcoef(projection1, projection2)[0, 1]
        
        # Label the plot
        ax_main.set_xlabel(f'Mode {mode_idx1} Amplitude', fontsize=11)  # Reduced from 12
        ax_main.set_ylabel(f'Mode {mode_idx2} Amplitude', fontsize=11)  # Reduced from 12
        ax_main.set_title(f'Phase Portrait: Mode {mode_idx1} vs Mode {mode_idx2}', 
                       fontweight='bold', fontsize=13)  # Reduced from 14
        
        # Add grid and equal aspect ratio for better visualization
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal', adjustable='box')
        
        # Add correlation and mode information
        ax_main.text(0.02, 0.98, f"Correlation: {corr_coef:.4f}", transform=ax_main.transAxes,
                  va='top', ha='left', fontsize=9,  # Reduced from 10
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
        
        # Add individual projections of each mode
        ax_time1 = fig.add_subplot(gs[1, 0])
        ax_time1.plot(projection1, 'o-', color='#1f77b4', markersize=3)  # Reduced marker size from 4
        ax_time1.set_title(f'Mode {mode_idx1} Projection', fontsize=10)
        ax_time1.set_xlabel('Time Steps', fontsize=9)
        ax_time1.set_ylabel('Amplitude', fontsize=9)
        ax_time1.grid(True, alpha=0.3)
        
        # Add individual projections of second mode
        ax_time2 = fig.add_subplot(gs[0, 1])
        ax_time2.plot(projection2, 'o-', color='#ff7f0e', markersize=3)  # Reduced marker size
        ax_time2.set_title(f'Mode {mode_idx2} Projection', fontsize=10)
        ax_time2.set_ylabel('Amplitude', fontsize=9)
        ax_time2.grid(True, alpha=0.3)
        # Rotate y-axis labels for better fit
        plt.setp(ax_time2.get_yticklabels(), rotation=90)
        
        # Add explanation
        ax_explain = fig.add_subplot(gs[1, 1])
        ax_explain.axis('off')
        explanation = "Phase Portrait: Shows relationship between modes. Spirals indicate growth/decay. Closed orbits = periodic behavior."
        ax_explain.text(0.5, 0.5, explanation, ha='center', va='center', fontsize=8,  # Reduced fontsize
                      wrap=True, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        # Increase spacing between subplots
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_reconstruction_error(self, figsize=(14, 8), save_path=None):
        """
        Plot the reconstruction error over time.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Get original and reconstructed data
        original = self.dmd.data
        reconstructed = self.dmd.reconstruct()
        
        if np.isreal(original).all():
            reconstructed = reconstructed.real
        
        # Calculate error over time
        error_over_time = np.linalg.norm(original - reconstructed, axis=0) / np.linalg.norm(original, axis=0)
        
        # Time array
        t = np.arange(error_over_time.shape[0]) * self.dmd.dt
        
        fig = plt.figure(figsize=figsize)
        
        # Plot error over time
        plt.plot(t, error_over_time, 'o-', markersize=4)  # Reduced marker size
        plt.xlabel('Time')
        plt.ylabel('Relative Error')
        plt.title('DMD Reconstruction Error Over Time')
        plt.grid(True, alpha=0.3)  # Reduced grid opacity
        
        # Add horizontal line for average error
        avg_error = np.mean(error_over_time)
        plt.axhline(y=avg_error, color='r', linestyle='--', 
                   label=f'Average Error: {avg_error:.4f}')
        plt.legend(fontsize=10)  # Added fontsize
        
        # Add more padding
        plt.tight_layout(pad=1.0)
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_mode_contributions(self, n_modes=8, figsize=(14, 12), save_path=None):
        """
        Plot the relative contributions of each DMD mode to the overall dynamics.
        
        Parameters:
        -----------
        n_modes : int, optional
            Number of modes to include in the plot. Reduced default from 10 to 8.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        amplitudes = np.abs(self.dmd.alpha)
        modes = self.dmd.phi
        n_features = modes.shape[0]
        
        # Sort modes by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        
        # Select top n_modes
        if n_modes > len(sorted_indices):
            n_modes = len(sorted_indices)
        
        mode_indices = sorted_indices[:n_modes]
        
        fig = plt.figure(figsize=figsize)
        
        # Create a grid of subplots
        n_cols = 2
        n_rows = int(np.ceil(n_modes / n_cols))
        
        # Create a more efficient subplot arrangement
        for i, mode_idx in enumerate(mode_indices):
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            mode_contribution = np.abs(modes[:, mode_idx] * self.dmd.alpha[mode_idx])
            
            # Normalize for visualization
            if np.max(mode_contribution) > 0:
                mode_contribution = mode_contribution / np.max(mode_contribution)
                
            ax.plot(range(n_features), mode_contribution, linewidth=1.5)  # Reduced linewidth
            
            # Compact title
            ax.set_title(f'Mode {mode_idx} (Amp: {amplitudes[mode_idx]:.3f})', fontsize=10)  # Reduced fontsize
            
            # Only show x label on bottom row
            if i >= n_modes - n_cols:
                ax.set_xlabel('Spatial Index', fontsize=9)
            
            # Only show y label on left side
            if i % n_cols == 0:
                ax.set_ylabel('Norm. Contribution', fontsize=9)
            
            ax.grid(True, alpha=0.3)  # Reduce grid opacity
            
            # Reduce tick label size
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Add spacing between subplots
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.35, wspace=0.25)
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_mode_frequencies(self, figsize=(12, 8), n_modes=None, save_path=None):
        """
        Plot the frequencies of DMD modes.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        n_modes : int, optional
            Number of modes to plot. If None, all modes are plotted.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Calculate frequencies (imaginary part of log(eigenvalues)/dt)
        dt = self.dmd.dt
        frequencies = np.imag(np.log(self.dmd.eigenvalues)) / (2 * np.pi * dt)
        amplitudes = np.abs(self.dmd.alpha)
        
        if n_modes is None:
            n_modes = len(frequencies)
        
        # Sort modes by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        sorted_frequencies = frequencies[sorted_indices]
        sorted_amplitudes = amplitudes[sorted_indices]
        
        fig = plt.figure(figsize=figsize)
        
        # Plot frequencies colored by amplitude - reduced marker size
        scatter = plt.scatter(range(min(n_modes, len(sorted_frequencies))), sorted_frequencies[:n_modes], 
                c=sorted_amplitudes[:n_modes], s=80, cmap='viridis')
    
        plt.xlabel('Mode Index (sorted by amplitude)')
        plt.ylabel('Frequency (Hz)')
        plt.title('DMD Mode Frequencies')
        plt.grid(True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Amplitude')
        
        # DO NOT use tight_layout() or plt.tight_layout(pad=1.0)
        # Instead, use a normal figure adjustment:
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
        return fig
    
    def plot_growth_rates(self, figsize=(12, 8), n_modes=None, save_path=None):
        """
        Plot the growth rates of DMD modes.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        n_modes : int, optional
            Number of modes to plot. If None, all modes are plotted.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Calculate growth rates (real part of log(eigenvalues)/dt)
        dt = self.dmd.dt
        growth_rates = np.real(np.log(self.dmd.eigenvalues)) / dt
        amplitudes = np.abs(self.dmd.alpha)
        
        if n_modes is None:
            n_modes = len(growth_rates)
        
        # Sort modes by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        sorted_growth_rates = growth_rates[sorted_indices]
        sorted_amplitudes = amplitudes[sorted_indices]
        
        fig = plt.figure(figsize=figsize)
        
        # Plot growth rates colored by amplitude - reduced marker size
        plt.scatter(range(min(n_modes, len(sorted_growth_rates))), sorted_growth_rates[:n_modes], 
                   c=sorted_amplitudes[:n_modes], s=80, cmap='viridis')  # Reduced from 100
        
        # Add horizontal line at y=0 to indicate stability boundary
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Mode Index (sorted by amplitude)')
        plt.ylabel('Growth Rate')
        plt.title('DMD Mode Growth Rates')
        plt.grid(True)
        plt.colorbar(label='Amplitude')
        
        # Add more spacing
        plt.tight_layout(pad=1.0)
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_mode_amplitudes(self, figsize=(14, 12), n_modes=None, save_path=None):
        """
        Plot the absolute amplitudes of DMD modes.
        
        This visualization shows the relative importance of each mode in the system:
        1. Higher amplitude modes have greater contribution to the overall dynamics
        2. The amplitude distribution indicates how many modes are needed to capture the system
        3. Sharp decline in amplitudes suggests the system can be well-approximated by few modes
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        n_modes : int, optional
            Number of modes to plot. If None, all modes are plotted.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        amplitudes = np.abs(self.dmd.alpha)
        
        if n_modes is None:
            n_modes = len(amplitudes)
        else:
            n_modes = min(n_modes, len(amplitudes))
        
        # Sort modes by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        sorted_amplitudes = amplitudes[sorted_indices]
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], figure=fig)
        
        # Main plot - amplitude bar chart
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Create bar chart with gradient colors
        bars = ax_main.bar(range(n_modes), sorted_amplitudes[:n_modes], width=0.7)
        
        # Color bars by relative amplitude
        if n_modes > 0:
            normalized_amplitudes = sorted_amplitudes[:n_modes] / sorted_amplitudes[0] if sorted_amplitudes[0] > 0 else sorted_amplitudes[:n_modes]
            cmap = cm.get_cmap('viridis')
            for i, bar in enumerate(bars):
                bar.set_color(cmap(normalized_amplitudes[i]))
                bar.set_edgecolor('black')
                bar.set_linewidth(0.5)
        
        # Add threshold line for 90% energy
        cumulative_energy = np.cumsum(sorted_amplitudes**2) / np.sum(sorted_amplitudes**2)
        rank_90 = np.where(cumulative_energy >= 0.9)[0][0] + 1 if len(cumulative_energy) > 0 else 0
        
        if rank_90 > 0 and rank_90 <= n_modes:
            ax_main.axvline(x=rank_90-0.5, color='r', linestyle='--', linewidth=2, 
                          label=f'90% Energy: {rank_90} modes')
        
        # Annotate top modes - only top 3 instead of 5 to reduce clutter
        for i in range(min(3, n_modes)):
            orig_idx = sorted_indices[i]
            ax_main.annotate(f'M{orig_idx}', # Shortened from "Mode"
                           xy=(i, sorted_amplitudes[i]),
                           xytext=(0, 5), textcoords='offset points',  # Reduced y-offset from 10
                           ha='center', va='bottom', fontsize=8,  # Added smaller fontsize
                           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8))
        
        ax_main.set_xlabel('Mode Rank (sorted by amplitude)', fontsize=11)  # Reduced from 12
        ax_main.set_ylabel('Absolute Amplitude', fontsize=11)  # Reduced from 12
        ax_main.set_title('DMD Mode Amplitudes (Ranked)', fontweight='bold', fontsize=13)  # Reduced from 14
        ax_main.grid(True, axis='y', alpha=0.3)
        ax_main.set_xlim(-0.5, n_modes-0.5)
        ax_main.legend(loc='best', fontsize=9)  # Reduced font size
        
        # Show original mode indices on secondary x-axis
        ax_main.set_xticks(range(n_modes))
        ax_main.set_xticklabels([f"{i+1}" for i in range(n_modes)])
        
        ax2 = ax_main.twiny()
        ax2.set_xlim(ax_main.get_xlim())
        ax2.set_xticks(range(n_modes))
        ax2.set_xticklabels([f"#{sorted_indices[i]}" for i in range(n_modes)])
        ax2.set_xlabel('Original Mode Index', fontsize=11)  # Reduced from 12
        
        # Add cumulative energy plot
        ax_cumulative = fig.add_subplot(gs[0, 1])
        ax_cumulative.plot(range(1, n_modes+1), cumulative_energy[:n_modes], 'o-', 
                          color='#1f77b4', linewidth=2, markersize=5)  # Reduced from 6
        
        # Add threshold lines - more compact annotations
        for threshold in [0.9, 0.95, 0.99]:
            if len(cumulative_energy) > 0:
                rank_idx = np.where(cumulative_energy >= threshold)[0][0] if np.any(cumulative_energy >= threshold) else len(cumulative_energy)-1
                ax_cumulative.axhline(y=threshold, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5)
                ax_cumulative.axvline(x=rank_idx+1, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5)
                ax_cumulative.annotate(f'{threshold*100:.0f}%: r{rank_idx+1}',  # Shortened
                                     xy=(rank_idx+1, threshold),
                                     xytext=(5, -5), textcoords='offset points',  # Reduced offset
                                     arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1),
                                     bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8),
                                     fontsize=8)  # Added smaller fontsize
        
        ax_cumulative.set_xlabel('Number of Modes')
        ax_cumulative.set_ylabel('Cumulative Energy')
        ax_cumulative.set_title('Cumulative Energy vs. Number of Modes')
        ax_cumulative.grid(True, alpha=0.3)
        ax_cumulative.set_xlim(0.5, n_modes+0.5)
        ax_cumulative.set_ylim(0, 1.05)
        
        # Add logarithmic amplitude plot
        ax_log = fig.add_subplot(gs[1, 0])
        ax_log.semilogy(range(1, n_modes+1), sorted_amplitudes[:n_modes], 'o-', 
                      color='#1f77b4', linewidth=2, markersize=5)  # Reduced from 6
        
        ax_log.set_xlabel('Mode Rank')
        ax_log.set_ylabel('Amplitude (log scale)')
        ax_log.set_title('Mode Amplitudes (Log Scale)')
        ax_log.grid(True, alpha=0.3)
        ax_log.set_xlim(0.5, n_modes+0.5)
        
        # Add explanation - more compact
        ax_explain = fig.add_subplot(gs[1, 1])
        ax_explain.axis('off')
        explanation = """Mode Amplitudes:
        • Bars: DMD modes by amplitude (importance)
        • Larger amplitudes: stronger contribution
        • Red line: modes needed for 90% energy
        • Steep drop-off: system is low-dimensional
        • Original mode indices on top x-axis"""
        
        ax_explain.text(0, 1, explanation, fontsize=9,  # Reduced from 12
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        # Add more space between subplots
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_complex_alpha(self, figsize=(10, 8), annotate=True, save_path=None):
        """
        Plot the complex amplitudes (alpha values) for DMD modes.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        annotate : bool, optional
            Whether to annotate points with mode indices.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig = plt.figure(figsize=figsize)
        
        # Plot complex amplitudes
        plt.scatter(self.dmd.alpha.real, self.dmd.alpha.imag, s=80, alpha=0.8)  # Reduced from 100
        
        if annotate:
            for i, alpha in enumerate(self.dmd.alpha):
                plt.annotate(str(i), xy=(alpha.real, alpha.imag),
                            xytext=(5, 5), textcoords='offset points', fontsize=9)  # Added fontsize
        
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Complex DMD Amplitudes (α)')
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig
    
    def plot_svd_analysis(self, figsize=(14, 10), save_path=None):
        """
        Plot singular value analysis from the DMD model.
        
        This plot helps determine the optimal truncation rank for DMD by showing:
        1. Singular value magnitudes - indicating the importance of each mode
        2. Cumulative energy - showing how much system dynamics are captured at each rank
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], figure=fig)
        
        # Plot singular values
        ax1 = fig.add_subplot(gs[0, 0])
        singular_values = self.dmd.singular_values
        indices = np.arange(1, len(singular_values) + 1)
        
        # Plot singular values with logarithmic scale
        ax1.semilogy(indices, singular_values, 'o-', color='#1f77b4', 
                   markersize=6, linewidth=2)  # Reduced marker size from 8
        
        # Add markers for potential truncation points
        for rank_threshold in [5, 10, 15]:
            if rank_threshold < len(singular_values):
                ax1.axvline(x=rank_threshold, color='#ff7f0e', linestyle='--', alpha=0.7, 
                          linewidth=1.5, label=f'r={rank_threshold}' if rank_threshold == 5 else None)
        
        # Add shaded region for noise floor (estimated as median of smallest singular values)
        if len(singular_values) > 10:
            noise_level = np.median(singular_values[-5:])
            ax1.axhline(y=noise_level, color='#d62728', linestyle=':', 
                      linewidth=1.5, label='Est. noise level')
            ax1.fill_between(indices, noise_level, min(singular_values), 
                           color='#d62728', alpha=0.1)
        
        ax1.set_xlabel('Mode Index')
        ax1.set_ylabel('Singular Value (log scale)')
        ax1.set_title('Singular Value Spectrum', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=9)  # Reduced font size
        
        # Add decay rate annotation
        if len(singular_values) > 2:
            decay_rate = singular_values[0] / singular_values[-1]
            ax1.annotate(f'Spectrum decay rate: {decay_rate:.1e}',
                       xy=(0.05, 0.05), xycoords='axes fraction',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                       fontsize=9)  # Added fontsize
        
        # Plot singular value gaps
        ax3 = fig.add_subplot(gs[1, 0])
        if len(singular_values) > 1:
            gaps = singular_values[:-1] / singular_values[1:]
            ax3.plot(indices[:-1], gaps, 'o-', color='#2ca02c', linewidth=2, markersize=5)  # Reduced marker size
            
            # Find maximum gap
            max_gap_idx = np.argmax(gaps)
            ax3.plot(indices[max_gap_idx], gaps[max_gap_idx], 'o', 
                   color='red', markersize=8, label='Max gap')  # Reduced marker size from 10
            
            ax3.set_xlabel('Mode Index')
            ax3.set_ylabel('sigma_i/sigma_i+1')
            ax3.set_title('Singular Value Gaps (larger gaps suggest truncation points)')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=9)  # Reduced font size
        
        # Plot cumulative energy
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        ax2.plot(indices, cumulative_energy, 'o-', color='#1f77b4', 
                markersize=6, linewidth=2)  # Reduced marker size from 8
        
        # Add threshold lines for common energy levels
        for threshold in [0.9, 0.95, 0.99]:
            rank_idx = np.where(cumulative_energy >= threshold)[0][0]
            ax2.axhline(y=threshold, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5)
            ax2.axvline(x=rank_idx + 1, color='#ff7f0e', linestyle='--', alpha=0.7, linewidth=1.5)
            # Make annotation more compact
            ax2.annotate(f'{threshold*100:.0f}%: rank {rank_idx + 1}', 
                        xy=(rank_idx + 1, threshold),
                        xytext=(5, -10), textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                        fontsize=8)  # Smaller font size
        
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Cumulative Energy')
        ax2.set_title('Cumulative Energy vs. Rank', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add explanation text - more compact version
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        explanation = """SVD Analysis:
        • Singular Value Spectrum: Importance of each mode
        • Cumulative Energy: % of dynamics captured at each rank
        • Singular Value Gaps: Natural truncation points
        • Choose rank that balances complexity vs accuracy"""
        
        ax4.text(0, 1, explanation, fontsize=9,  # Reduced from 11
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        # Use tight_layout() instead of subplots_adjust at the end
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_spectrum(self, figsize=(14, 12), save_path=None):
        """
        Plot the DMD spectrum (eigenvalues on the complex plane).
        
        This plot shows eigenvalues in the complex plane, which reveals:
        1. Stability: Points outside the unit circle represent growing/unstable modes
        2. Frequencies: Angular position corresponds to mode frequency
        3. Mode importance: Color/size represents mode amplitude/energy
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1], figure=fig)
        
        # Main plot - eigenvalues on complex plane
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Normalize amplitudes for better visualization
        amplitudes = np.abs(self.dmd.alpha)
        norm_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        # Plot eigenvalues with size and color based on amplitudes - reduced sizes
        scatter = ax_main.scatter(self.dmd.eigenvalues.real, self.dmd.eigenvalues.imag, 
                               c=norm_amplitudes, s=200*norm_amplitudes + 30,  # Reduced from 300/50
                               cmap='plasma', alpha=0.8, edgecolor='k', linewidth=0.5)
        
        # Add unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax_main.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.7, linewidth=2, 
                   label='Unit Circle (Stability Boundary)')
        
        # Add axes lines
        ax_main.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax_main.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Label the top modes
        top_n = 3  # Number of top modes to label
        sorted_indices = np.argsort(norm_amplitudes)[::-1][:top_n]
        
        for i, idx in enumerate(sorted_indices):
            eig = self.dmd.eigenvalues[idx]
            ax_main.annotate(f'Mode {idx}', xy=(eig.real, eig.imag),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                           arrowprops=dict(arrowstyle="->"),
                           fontsize=9)  # Added fontsize
        
        ax_main.set_xlabel('Real Part')
        ax_main.set_ylabel('Imaginary Part')
        ax_main.set_title('DMD Spectrum (Eigenvalues)', fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.axis('equal')
        ax_main.legend(loc='best', fontsize=9)  # Reduced font size
        
        # Add color bar
        cbar = plt.colorbar(scatter, ax=ax_main)
        cbar.set_label('Normalized Mode Amplitude', fontsize=10)  # Added fontsize
        
        # Add stability analysis subplot
        ax_stability = fig.add_subplot(gs[0, 1])
        
        # Calculate magnitudes
        magnitudes = np.abs(self.dmd.eigenvalues)
        
        # Create histogram of magnitudes
        ax_stability.hist(magnitudes, bins=15, orientation='horizontal', alpha=0.7, color='#1f77b4')
        ax_stability.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Stability Limit')
        
        # Calculate stability statistics
        n_unstable = np.sum(magnitudes > 1.001)  # Add small buffer for numerical error
        n_marginal = np.sum((magnitudes <= 1.001) & (magnitudes >= 0.999))
        n_stable = np.sum(magnitudes < 0.999)
        
        # Add text annotations - more compact
        ax_stability.text(0.5, 0.9, f"Unstable: {n_unstable}", transform=ax_stability.transAxes,
                         ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", fc="#ffcccc", ec="gray", alpha=0.8))
        ax_stability.text(0.5, 0.8, f"Marginal: {n_marginal}", transform=ax_stability.transAxes,
                         ha='center', va='center', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="gray", alpha=0.8))
        ax_stability.text(0.5, 0.7, f"Stable: {n_stable}", transform=ax_stability.transAxes,
                         ha='center', va='center', fontsize=9, 
                         bbox=dict(boxstyle="round,pad=0.2", fc="#ccffcc", ec="gray", alpha=0.8))
        
        ax_stability.set_ylabel('|λ| (Magnitude)')
        ax_stability.set_xlabel('Count')
        ax_stability.set_title('Stability Analysis')
        
        # Add explanation text - more compact
        ax_explain = fig.add_subplot(gs[1, :])
        ax_explain.axis('off')
        explanation = """DMD Spectrum Explained:
        • Each point: DMD mode with eigenvalue (λ) in complex plane
        • Real part (x): Growth/decay rate. Positive values = growth
        • Imaginary part (y): Oscillation frequency
        • Unit circle: Stability boundary. Outside = unstable modes
        • Point size & color: Mode amplitude/importance
        • Real eigenvalues (x-axis): Non-oscillatory modes
        • Complex pairs: Oscillatory modes (symmetric across x-axis)"""
        
        ax_explain.text(0, 1, explanation, fontsize=10,  # Reduced from 12
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        # Increase spacing between subplots
        fig.subplots_adjust(hspace=0.35, wspace=0.30)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_continuous_spectrum(self, figsize=(10, 8), save_path=None):
        """
        Plot the continuous-time DMD spectrum.
        
        This plot shows the continuous-time eigenvalues, which provides:
        1. Growth/decay rates: Real part directly shows exponential growth/decay rates
        2. Frequencies: Imaginary part directly shows oscillation frequencies in rad/time unit
        3. Stability boundary: Vertical line at x=0 separates stable (left) from unstable (right) modes
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Calculate continuous-time eigenvalues
        dt = self.dmd.dt
        continuous_evals = np.log(self.dmd.eigenvalues) / dt
        
        # Normalize amplitudes for better visualization
        amplitudes = np.abs(self.dmd.alpha)
        norm_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[10, 1, 3], figure=fig)
        
        # Main plot - continuous eigenvalues
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Create scatter plot with size and color based on amplitudes - reduced sizes
        scatter = ax_main.scatter(continuous_evals.real, continuous_evals.imag, 
                               c=norm_amplitudes, s=200*norm_amplitudes + 30,  # Reduced from 300/50
                               cmap='plasma', alpha=0.8, edgecolor='k', linewidth=0.5)
        
        # Add vertical line for stability boundary
        ax_main.axvline(x=0, color='r', linestyle='--', alpha=0.7, linewidth=2, 
                      label='Stability Boundary')
        
        # Add horizontal line for zero frequency
        ax_main.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Label frequencies on y-axis in Hz
        y_ticks = ax_main.get_yticks()
        ax_main.set_yticks(y_ticks)  # Set ticks first, then labels
        ax_main.set_yticklabels([f"{y/(2*np.pi):.2f}" for y in y_ticks])
        
        # Label the top modes - more compact labels
        top_n = 3  # Number of top modes to label
        sorted_indices = np.argsort(norm_amplitudes)[::-1][:top_n]
        
        for i, idx in enumerate(sorted_indices):
            eig = continuous_evals[idx]
            freq_hz = abs(eig.imag) / (2*np.pi)
            growth = eig.real
            
            # Create shorter label
            label = f"M{idx}: {freq_hz:.2f}Hz, {growth:.2f}gr"
            
            ax_main.annotate(label, xy=(eig.real, eig.imag),
                           xytext=(10, 0), textcoords='offset points',
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
                           arrowprops=dict(arrowstyle="->"),
                           fontsize=8)  # Smaller font
        
        ax_main.set_xlabel('Real Part (Growth/Decay Rate)', fontsize=11)  # Reduced from 12
        ax_main.set_ylabel('Imaginary Part (Angular Frequency)', fontsize=11)  # Reduced from 12
        ax_main.set_title('Continuous-Time DMD Spectrum', fontweight='bold', fontsize=13)  # Reduced from 14
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='best', fontsize=9)  # Reduced font size
        
        # Add secondary y-axis for frequency in Hz
        ax2 = ax_main.twinx()
        ax2.set_ylabel('Frequency (Hz)', fontsize=11)  # Reduced from 12
        
        # Set the same limits as the main y-axis
        ax2.set_ylim(ax_main.get_ylim())
        
        # Convert the ticks to Hz
        ax2_ticks = ax2.get_yticks()
        ax2.set_yticks(ax2_ticks)
        ax2.set_yticklabels([f"{y/(2*np.pi):.2f}" for y in ax2_ticks])
        
        # Add color bar
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = plt.colorbar(scatter, cax=ax_cbar)
        cbar.set_label('Normalized Mode Amplitude', fontsize=10)
        
        # Add growth rate histogram
        ax_growth = fig.add_subplot(gs[0, 2])
        growth_rates = continuous_evals.real
        
        ax_growth.hist(growth_rates, bins=15, orientation='horizontal', alpha=0.7, color='#1f77b4')
        ax_growth.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Stability Limit')
        
        # Calculate growth rate statistics
        n_growing = np.sum(growth_rates > 0.001)  # Add small buffer for numerical error
        n_neutral = np.sum((growth_rates <= 0.001) & (growth_rates >= -0.001))
        n_decaying = np.sum(growth_rates < -0.001)
        
        # Add text annotations - more compact
        ax_growth.text(0.5, 0.9, f"Growing: {n_growing}", transform=ax_growth.transAxes,
                      ha='center', va='center', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2", fc="#ffcccc", ec="gray", alpha=0.8))
        ax_growth.text(0.5, 0.8, f"Neutral: {n_neutral}", transform=ax_growth.transAxes,
                      ha='center', va='center', fontsize=9,
                      bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="gray", alpha=0.8))
        ax_growth.text(0.5, 0.7, f"Decaying: {n_decaying}", transform=ax_growth.transAxes,
                      ha='center', va='center', fontsize=9, 
                      bbox=dict(boxstyle="round,pad=0.2", fc="#ccffcc", ec="gray", alpha=0.8))
        
        ax_growth.set_ylabel('Growth Rate')
        ax_growth.set_xlabel('Count')
        ax_growth.set_title('Growth Rate Distribution')
        
        # Add explanation text - more compact
        ax_explain = fig.add_subplot(gs[1, :])
        ax_explain.axis('off')
        explanation = """Continuous-Time DMD Spectrum:
        • Shows DMD eigenvalues in continuous-time domain (log of discrete eigenvalues)
        • Real part (x): Exponential growth/decay rate. Positive = growing modes
        • Imaginary part (y): Oscillation frequency in rad/time (left axis) and Hz (right)
        • Red vertical line: Stability boundary. Right side = unstable modes
        • Point size & color: Mode amplitude/importance
        • Conversion: λc = log(λ)/dt where λ are discrete eigenvalues and dt is time step
        • Physical interpretation: Each mode evolves as exp(λc·t) where t is time"""
        
        ax_explain.text(0, 1, explanation, fontsize=10,  # Reduced from 12
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_3d_spectrum(self, figsize=(14, 14), save_path=None):
        """
        Plot a 3D representation of the DMD spectrum showing eigenvalues and amplitudes.
        
        This visualization combines eigenvalue positions with amplitudes in 3D space:
        1. Complex plane (x-y) shows eigenvalue positions as in the standard spectrum plot
        2. Vertical axis (z) shows the normalized amplitude of each mode
        3. Vertical lines connect eigenvalues to their projections on the complex plane
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        fig = plt.figure(figsize=figsize)
        
        # Create a GridSpec with 2 rows, the top row for 3D plot, bottom for explanation
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        
        # Create 3D axes
        ax = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Extract data
        real_parts = self.dmd.eigenvalues.real
        imag_parts = self.dmd.eigenvalues.imag
        amplitudes = np.abs(self.dmd.alpha)
        
        # Normalize amplitudes for better visualization
        norm_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        # Create colormap based on eigenvalue magnitude (stability)
        eigenvalue_mags = np.abs(self.dmd.eigenvalues)
        colors = plt.cm.RdYlGn_r(1.0 - np.minimum(eigenvalue_mags/1.5, 1.0))
        
        # Create 3D scatter plot - reduced sizes
        scatter = ax.scatter(real_parts, imag_parts, norm_amplitudes, 
                           c=colors, s=80*norm_amplitudes+40, alpha=0.8,  # Reduced from 100/50
                           edgecolor='k', linewidth=0.5)
        
        # Add unit circle on the complex plane
        theta = np.linspace(0, 2*np.pi, 100)
        x_circle = np.cos(theta)
        y_circle = np.sin(theta)
        z_circle = np.zeros(100)
        ax.plot(x_circle, y_circle, z_circle, 'k--', alpha=0.7, linewidth=2, label='Unit Circle')
        
        # Add vertical lines from points to the complex plane
        for i in range(len(real_parts)):
            ax.plot([real_parts[i], real_parts[i]], 
                   [imag_parts[i], imag_parts[i]], 
                   [0, norm_amplitudes[i]], 'k-', alpha=0.3)
        
        # Add coordinate axes on the complex plane
        ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'k-', alpha=0.5, linewidth=1)  # Real axis
        ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'k-', alpha=0.5, linewidth=1)  # Imaginary axis
        
        # Label the top modes
        top_n = 3  # Number of top modes to label
        sorted_indices = np.argsort(norm_amplitudes)[::-1][:top_n]
        
        for i, idx in enumerate(sorted_indices):
            ax.text(real_parts[idx], imag_parts[idx], norm_amplitudes[idx] + 0.05, 
                   f'Mode {idx}', fontsize=9, ha='center')  # Reduced from 10
        
        # Set labels and title
        ax.set_xlabel('Real Part', fontsize=11)  # Reduced from 12
        ax.set_ylabel('Imaginary Part', fontsize=11)  # Reduced from 12
        ax.set_zlabel('Normalized Amplitude', fontsize=11)  # Reduced from 12
        ax.set_title('3D DMD Spectrum (Eigenvalues and Amplitudes)', fontweight='bold', fontsize=13)  # Reduced from 14
        
        # Set axis limits for better visualization
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 1.1)
        
        # Add a legend
        ax.legend(loc='upper right', fontsize=9)  # Reduced font size
        
        # Add annotation for color meaning - more compact
        color_explanation = (
            "Color: Red=Unstable, Yellow=Marginal, Green=Stable"
        )
        ax.text2D(0.02, 0.95, color_explanation, transform=ax.transAxes,
                 fontsize=9, verticalalignment='top',  # Reduced from 10
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set optimal viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Add explanation in the bottom subplot - more compact
        ax_exp = fig.add_subplot(gs[1, 0])
        ax_exp.axis('off')
        
        explanation = """3D DMD Spectrum Explained:
        • Each point: DMD mode with eigenvalue on complex plane (x-y) and amplitude as height (z)
        • Points above unit circle (black circle) are unstable/growing modes
        • Tallest points: Most dominant modes in the system
        • Color: Red = Unstable, Yellow = Marginal, Green = Stable
        • Black vertical lines connect points to complex plane projections
        • This visualization shows both mode stability (position) and importance (height)"""
        
        ax_exp.text(0.5, 0.5, explanation, fontsize=10, ha='center', va='center',  # Reduced from 12
                  bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        # Adjust spacing
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

    def plot_forecast(self, spatial_point, future_times, future_states, confidence=None, figsize=(32, 16), save_path=None):
        """
        Plot forecast for a specific spatial point.
        
        This visualization shows how well DMD predicts future states:
        1. Original data vs reconstruction shows model accuracy on training data
        2. Forecast region shows predicted future states with confidence intervals
        3. Clear separation between training and prediction regions
        
        Parameters:
        -----------
        spatial_point : int
            Index of the spatial point to visualize.
        future_times : array
            Array of future time points.
        future_states : array
            Array of forecasted states.
        confidence : dict, optional
            Dictionary mapping time points to confidence levels.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        save_path : str, optional
            Path to save the figure.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object.
        """
        # Get original and reconstructed data
        original = self.dmd.data
        reconstructed = self.dmd.reconstruct()
        
        if np.isreal(original).all():
            reconstructed = reconstructed.real
            future_states = future_states.real
        
        # Time array for original data
        t = np.arange(original.shape[1]) * self.dmd.dt
        
        # Initialize figure
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], figure=fig)
        
        # Main plot
        ax_main = fig.add_subplot(gs[0, :])
        
        # Plot original data
        ax_main.plot(t, original[spatial_point, :], 'o-', color='#1f77b4', 
                   markersize=6, linewidth=2, label='Original Data')
        
        # Plot DMD reconstruction
        ax_main.plot(t, reconstructed[spatial_point, :], '-', color='#ff7f0e', 
                   linewidth=2, label='DMD Reconstruction')
        
        # Plot DMD forecast with different style in the prediction region
        ax_main.plot(future_times, future_states[spatial_point, :], '--', color='#2ca02c', 
                   linewidth=2, label='DMD Forecast')
        
        # Add confidence intervals if provided
        if confidence is not None:
            upper_bound = np.zeros_like(future_states[spatial_point, :])
            lower_bound = np.zeros_like(future_states[spatial_point, :])
            
            for i, ti in enumerate(future_times):
                conf = confidence.get(ti, 0.5)
                margin = (1.0 - conf) * np.std(original[spatial_point, :]) * 3
                upper_bound[i] = future_states[spatial_point, i] + margin
                lower_bound[i] = future_states[spatial_point, i] - margin
            
            ax_main.fill_between(future_times, lower_bound, upper_bound, 
                               color='#2ca02c', alpha=0.2, label='95% Confidence Interval')
        
        # Vertical line separating training from prediction
        training_end = t[-1]
        ax_main.axvline(x=training_end, color='red', linestyle='-', linewidth=2, 
                      label='End of Training Data')
        
        # Add shaded regions for training and prediction
        ax_main.axvspan(0, training_end, alpha=0.1, color='blue', label='Training Region')
        ax_main.axvspan(training_end, max(future_times), alpha=0.1, color='green', label='Prediction Region')
        
        # Calculate and display error metrics
        train_rmse = np.sqrt(np.mean((original[spatial_point, :] - reconstructed[spatial_point, :])**2))
        
        # Find future states that overlap with original data timepoints for validation
        common_times = []
        common_orig = []
        common_pred = []
        
        for i, ft in enumerate(future_times):
            if ft <= t[-1]:  # If this future time is within the original data range
                # Find the closest original time point
                closest_idx = np.argmin(np.abs(t - ft))
                if np.abs(t[closest_idx] - ft) < 1e-6:  # If they match closely
                    common_times.append(ft)
                    common_orig.append(original[spatial_point, closest_idx])
                    common_pred.append(future_states[spatial_point, i])
        
        val_rmse = 0
        if common_orig and common_pred:
            val_rmse = np.sqrt(np.mean(np.array(common_orig) - np.array(common_pred))**2)
        
        metrics_text = (
            f"Performance Metrics:\n"
            f"Training RMSE: {train_rmse:.4f}\n"
            f"Validation RMSE: {val_rmse:.4f}\n"
            f"Spatial Point: {spatial_point}/{original.shape[0]-1}"
        )
        
        # Position text box in the corner of the training region
        ax_main.text(0.02, 0.02, metrics_text, transform=ax_main.transAxes,
                   va='bottom', ha='left', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax_main.set_xlabel('Time', fontsize=12)
        ax_main.set_ylabel('Amplitude', fontsize=12)
        ax_main.set_title(f'DMD Reconstruction and Forecast at Spatial Point {spatial_point}', 
                        fontweight='bold', fontsize=14)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='best')
        
        # Add residual plot
        ax_residual = fig.add_subplot(gs[1, :])
        
        # Calculate residuals for the training period
        residuals = original[spatial_point, :] - reconstructed[spatial_point, :]
        
        # Plot residuals
        ax_residual.plot(t, residuals, 'o-', color='#d62728', markersize=4, linewidth=1)
        ax_residual.axhline(y=0, color='k', linestyle='-', linewidth=1)
        
        # Add reference lines for standard deviation
        std_dev = np.std(residuals)
        ax_residual.axhline(y=std_dev, color='k', linestyle='--', linewidth=1, alpha=0.5, 
                         label=f'±1σ (σ={std_dev:.4f})')
        ax_residual.axhline(y=-std_dev, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add vertical line separating training from prediction
        ax_residual.axvline(x=training_end, color='red', linestyle='-', linewidth=2)
        
        ax_residual.set_xlabel('Time', fontsize=12)
        ax_residual.set_ylabel('Residual\n(Original - Reconstructed)', fontsize=10)
        ax_residual.set_title('Reconstruction Residuals')
        ax_residual.grid(True, alpha=0.3)
        ax_residual.legend(loc='best')
        
        # Match x-axis limits with main plot
        ax_residual.set_xlim(ax_main.get_xlim())
        
        # Add explanation text as a figure title
        explanation = """
            DMD Forecast Visualization: Shows how well the DMD model captures system dynamics and predicts future states.
            Blue points/line: Original data | Orange line: DMD reconstruction | Green dashed line: DMD forecast
            The red vertical line separates training data (left) from prediction (right). Confidence intervals show prediction uncertainty.
            Residual plot shows reconstruction errors over time to identify systematic biases or patterns.
        """
        fig.suptitle(explanation, fontsize=14, y=0.99)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.85, hspace=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig