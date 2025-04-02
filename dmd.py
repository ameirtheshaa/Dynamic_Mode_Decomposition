import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import logm
import pandas as pd
import seaborn as sns
from IPython.display import display

class DMD:
    """
    Dynamic Mode Decomposition (DMD) class for analyzing dynamical systems.
    
    This implementation provides a comprehensive set of tools for DMD analysis including:
    - Standard DMD and exact DMD computation
    - Mode analysis and diagnostics
    - Future state prediction
    - Various visualization methods
    
    Parameters
    ----------
    data : numpy.ndarray
        The input data matrix. Each column represents a snapshot of the system at a given time.
    dt : float, optional
        Time step between snapshots, default is 1.0.
    rank : int or None, optional
        Truncation rank for SVD. If None, no truncation is performed.
    """
    
    def __init__(self, data, dt=15, rank=None):
        """
        Initialize the DMD object with data and optional parameters.
        
        Parameters
        ----------
        data : numpy.ndarray
            The input data matrix. Each column represents a snapshot of the system at a given time.
        dt : float, optional
            Time step between snapshots, default is 1.0.
        rank : int or None, optional
            Truncation rank for SVD. If None, no truncation is performed.
        """
        if len(data.shape) != 2:
            raise ValueError("Input data must be a 2D array")
        
        self.data = data
        self.dt = dt
        self.rank = rank
        
        # Store original dimensions
        self.n_features, self.n_snapshots = data.shape
        
        # Create snapshot matrices (X_1 is all columns except the last, X_2 is all columns except the first)
        self.X_1 = data[:, :-1]
        self.X_2 = data[:, 1:]
        
        # Compute DMD
        self._compute_dmd()
        
    def _compute_dmd(self):
        """Compute the Dynamic Mode Decomposition."""
        
        # SVD of X_1
        U, sigma, Vh = np.linalg.svd(self.X_1, full_matrices=False)
        V = Vh.T
        
        # Truncate if rank is specified
        if self.rank is not None:
            r = min(self.rank, len(sigma))
            U = U[:, :r]
            sigma = sigma[:r]
            V = V[:, :r]
        else:
            r = len(sigma)
        
        # Store rank that was used
        self.effective_rank = r
        
        # Create Sigma inverse
        Sigma_inv = np.diag(1.0 / sigma)
        
        # Compute A tilde (low-rank approximation of A)
        self.A_tilde = U.T @ self.X_2 @ V @ Sigma_inv
        
        # Eigendecomposition of A_tilde
        self.Lambda, self.W = np.linalg.eig(self.A_tilde)
        
        # Compute DMD modes
        self.phi = self.X_2 @ V @ Sigma_inv @ self.W
        
        # Compute initial amplitudes
        self.alpha = np.linalg.pinv(self.phi) @ self.X_1[:, 0]
        
        # Compute continuous-time eigenvalues (growth rates and frequencies)
        self.omega = np.log(self.Lambda) / self.dt
        
        # Compute DMD spectrum (magnitudes of continuous eigenvalues)
        self.dmd_spectrum = np.abs(self.Lambda)
        
        # Store matrices for diagnostics
        self.U = U
        self.sigma = sigma
        self.V = V
        self.Sigma_inv = Sigma_inv
        self.singular_values = sigma
        self.eigenvalues = self.Lambda
        
    def reconstruct(self, times=None):
        """
        Reconstruct the data using the DMD modes.
        
        Parameters
        ----------
        times : array-like, optional
            Times at which to reconstruct the data. If None, uses the original snapshot times.
            
        Returns
        -------
        numpy.ndarray
            Reconstructed data at specified times.
        """
        if times is None:
            times = np.arange(self.n_snapshots) * self.dt
        
        # Create time dynamics matrix
        time_dynamics = np.zeros((len(self.Lambda), len(times)), dtype=complex)
        for i, t in enumerate(times):
            time_dynamics[:, i] = np.exp(self.omega * t)
        
        # Reconstruct data
        return self.phi @ np.diag(self.alpha) @ time_dynamics
    
    def predict(self, times):
        """
        Predict future states of the system.
        
        Parameters
        ----------
        times : array-like
            Times at which to predict the state of the system.
            
        Returns
        -------
        numpy.ndarray
            Predicted states at specified times.
        """
        return self.reconstruct(times)
    
    def mode_frequencies(self):
        """
        Calculate the frequency and growth rate of each mode.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing mode index, frequency, growth rate, 
            magnitude, and normalized amplitude.
        """
        frequencies = np.imag(self.omega) / (2 * np.pi)
        growth_rates = np.real(self.omega)
        magnitudes = np.abs(self.Lambda)
        amplitudes = np.abs(self.alpha)
        self.amplitudes = amplitudes
        norm_amplitudes = amplitudes / np.sum(amplitudes)
        
        data = {
            'Mode': np.arange(len(self.Lambda)),
            'Frequency': frequencies,
            'Growth_Rate': growth_rates,
            'Magnitude': magnitudes,
            'Amplitude': amplitudes,
            'Normalized_Amplitude': norm_amplitudes
        }
        
        return pd.DataFrame(data).sort_values(by='Normalized_Amplitude', ascending=False)
    
    def mode_significance(self):
        """
        Calculate the significance of each mode based on its amplitude and growth/decay.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing modes sorted by their significance.
        """
        # Calculate the significance as a combination of amplitude and magnitude
        amplitudes = np.abs(self.alpha)
        magnitudes = np.abs(self.Lambda)
        
        # Normalize both to [0,1]
        norm_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        # Calculate significance score (simple product)
        significance = norm_amplitudes * magnitudes
        
        data = {
            'Mode': np.arange(len(self.Lambda)),
            'Significance': significance,
            'Normalized_Amplitude': norm_amplitudes,
            'Magnitude': magnitudes
        }
        
        return pd.DataFrame(data).sort_values(by='Significance', ascending=False)
    
    def residual_analysis(self):
        """
        Perform residual analysis to assess the quality of the DMD approximation.
        
        Returns
        -------
        dict
            Dictionary containing residual statistics.
        """
        # Reconstruct the data
        reconstructed = self.reconstruct()
        
        # For real-valued data, take the real part
        if np.isreal(self.data).all():
            reconstructed = reconstructed.real
        
        # Calculate residuals
        residuals = self.data - reconstructed
        
        # Calculate error metrics
        relative_error = np.linalg.norm(residuals) / np.linalg.norm(self.data)
        max_error = np.max(np.abs(residuals))
        mean_error = np.mean(np.abs(residuals))
        
        return {
            'relative_error': relative_error,
            'max_error': max_error,
            'mean_error': mean_error,
            'residuals': residuals
        }
    
    def eigenvalue_check(self):
        """
        Check the validity of DMD eigenvalues.
        
        Returns
        -------
        float
            Spectral radius of DMD operator A.
        """
        # Check if any eigenvalues have magnitude > 1 (indicating growth)
        growing_modes = np.sum(np.abs(self.Lambda) > 1.0)
        print(f"Number of growing modes (|λ| > 1): {growing_modes}")
        
        # Check spectral radius
        spectral_radius = np.max(np.abs(self.Lambda))
        print(f"Spectral radius: {spectral_radius}")
        
        return spectral_radius
    
    def modes_orthogonality(self):
        """
        Check the orthogonality of the DMD modes without storing the full Gram matrix.
        
        Returns
        -------
        float
            Maximum off-diagonal element in the Gram matrix.
        float
            Deviation from orthonormality.
        """
        n = self.phi.shape[1]  # Number of modes
        
        # Initialize tracking variables
        max_off_diag = 0.0
        sum_squared_diff = 0.0
        
        # Process the Gram matrix one column at a time
        for i in range(n):
            # Compute one column of the Gram matrix
            col_i = self.phi.T.conj()[:, i]  # i-th row of phi.T.conj()
            gram_col = self.phi @ col_i
            
            # Check off-diagonal elements in this column
            for j in range(n):
                if i == j:
                    # For diagonal elements, compute deviation from 1
                    sum_squared_diff += abs(gram_col[j] - 1.0)**2
                else:
                    # For off-diagonal elements, update max and add to sum
                    max_off_diag = max(max_off_diag, abs(gram_col[j]))
                    sum_squared_diff += abs(gram_col[j])**2
        
        deviation = np.sqrt(sum_squared_diff)
        
        print(f"Maximum off-diagonal element in Gram matrix: {max_off_diag}")
        print(f"Deviation from orthonormality: {deviation}")
        
        return max_off_diag, deviation
    
    def spatial_mode_visualization(self, mode_indices=None, figsize=(12, 8), grid_size=None):
        """
        Visualize the spatial structure of selected DMD modes.
        
        Parameters
        ----------
        mode_indices : list, optional
            Indices of modes to visualize. If None, visualizes the top modes by amplitude.
        figsize : tuple, optional
            Figure size.
        grid_size : tuple, optional
            Grid size for reshaping the spatial modes (for 2D data).
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the visualizations.
        """
        if mode_indices is None:
            # Get the top modes by amplitude
            mode_indices = np.argsort(np.abs(self.alpha))[-4:][::-1]
        
        n_modes = len(mode_indices)
        fig, axes = plt.subplots(n_modes, 2, figsize=figsize)
        
        if n_modes == 1:
            axes = axes.reshape(1, 2)
        
        for i, mode_idx in enumerate(mode_indices):
            mode = self.phi[:, mode_idx]
            
            # Real part
            ax = axes[i, 0]
            if grid_size is not None:
                ax.imshow(np.real(mode).reshape(grid_size), cmap='RdBu_r')
                ax.set_title(f'Mode {mode_idx} (Real)')
            else:
                ax.plot(np.real(mode))
                ax.set_title(f'Mode {mode_idx} (Real Part)')
                ax.grid(True)
            
            # Imaginary part
            ax = axes[i, 1]
            if grid_size is not None:
                ax.imshow(np.imag(mode).reshape(grid_size), cmap='RdBu_r')
                ax.set_title(f'Mode {mode_idx} (Imaginary)')
            else:
                ax.plot(np.imag(mode))
                ax.set_title(f'Mode {mode_idx} (Imaginary Part)')
                ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_spectrum(self, figsize=(10, 8), size_factor=100):
        """
        Plot the DMD spectrum in the complex plane.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        size_factor : float, optional
            Scaling factor for the size of the points.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the spectrum plot.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.6)
        
        # Plot eigenvalues
        amplitudes = np.abs(self.alpha)
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        scatter = ax.scatter(self.Lambda.real, self.Lambda.imag, 
                             s=normalized_amplitudes*size_factor, 
                             c=np.abs(self.Lambda), 
                             cmap='viridis', 
                             alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnitude |λ|')
        
        # Set labels and title
        ax.set_xlabel('Real(λ)')
        ax.set_ylabel('Imag(λ)')
        ax.set_title('DMD Spectrum (Discrete Eigenvalues)')
        
        # Add grid and equal aspect ratio
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.annotate(f'{i}', 
                        (self.Lambda[i].real, self.Lambda[i].imag),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10)
        
        return fig
    
    def plot_continuous_spectrum(self, figsize=(10, 8), size_factor=100):
        """
        Plot the continuous-time DMD spectrum.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        size_factor : float, optional
            Scaling factor for the size of the points.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the continuous-time spectrum plot.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot eigenvalues
        amplitudes = np.abs(self.alpha)
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        scatter = ax.scatter(self.omega.real, self.omega.imag, 
                             s=normalized_amplitudes*size_factor, 
                             c=np.abs(self.omega), 
                             cmap='plasma', 
                             alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Magnitude |ω|')
        
        # Set labels and title
        ax.set_xlabel('Growth Rate (Real(ω))')
        ax.set_ylabel('Frequency (Imag(ω))')
        ax.set_title('DMD Continuous-Time Spectrum')
        
        # Add grid
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.annotate(f'{i}', 
                        (self.omega[i].real, self.omega[i].imag),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=10)
        
        return fig
    
    def plot_mode_amplitudes(self, figsize=(10, 6), n_modes=None):
        """
        Plot the amplitudes of the DMD modes.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the amplitude plot.
        """
        amplitudes = np.abs(self.alpha)
        
        if n_modes is None:
            n_modes = len(amplitudes)
        else:
            n_modes = min(n_modes, len(amplitudes))
        
        # Sort indices by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1][:n_modes]
        
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(np.arange(n_modes), amplitudes[sorted_indices])
        
        # Color bars by eigenvalue magnitude
        magnitudes = np.abs(self.Lambda[sorted_indices])
        normalized_magnitudes = magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes
        
        # Create colormap
        cmap = plt.cm.viridis
        
        for i, bar in enumerate(bars):
            bar.set_color(cmap(normalized_magnitudes[i]))
        
        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Amplitude')
        ax.set_title('DMD Mode Amplitudes')
        
        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_mode_frequencies(self, figsize=(10, 6), n_modes=None):
        """
        Plot the frequencies of the DMD modes.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the frequency plot.
        """
        frequencies = np.abs(np.imag(self.omega) / (2 * np.pi))  # Convert to cycles per time unit
        amplitudes = np.abs(self.alpha)
        
        # Sort by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        
        if n_modes is None:
            n_modes = len(frequencies)
        else:
            n_modes = min(n_modes, len(frequencies))
            
        sorted_indices = sorted_indices[:n_modes]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create scatter plot with size proportional to amplitude
        normalized_amplitudes = amplitudes[sorted_indices] / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes[sorted_indices]
        
        scatter = ax.scatter(sorted_indices, frequencies[sorted_indices], 
                             s=normalized_amplitudes*200, 
                             c=normalized_amplitudes, 
                             cmap='viridis', 
                             alpha=0.7)
        
        # Connect points with line
        ax.plot(np.arange(n_modes), frequencies[sorted_indices], 'k-', alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Amplitude')
        
        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Frequency (cycles per time unit)')
        ax.set_title('DMD Mode Frequencies')
        
        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_growth_rates(self, figsize=(10, 6), n_modes=None):
        """
        Plot the growth rates of the DMD modes.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        n_modes : int, optional
            Number of modes to plot. If None, plots all modes.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the growth rate plot.
        """
        growth_rates = np.real(self.omega)
        amplitudes = np.abs(self.alpha)
        
        # Sort by amplitude
        sorted_indices = np.argsort(amplitudes)[::-1]
        
        if n_modes is None:
            n_modes = len(growth_rates)
        else:
            n_modes = min(n_modes, len(growth_rates))
            
        sorted_indices = sorted_indices[:n_modes]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar chart
        bars = ax.bar(np.arange(n_modes), growth_rates[sorted_indices])
        
        # Color bars by sign of growth rate
        for i, bar in enumerate(bars):
            if growth_rates[sorted_indices[i]] >= 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        # Set labels and title
        ax.set_xlabel('Mode Index (sorted by amplitude)')
        ax.set_ylabel('Growth Rate')
        ax.set_title('DMD Mode Growth Rates')
        
        # Add horizontal line at y=0
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        # Add mode indices as x-tick labels
        ax.set_xticks(np.arange(n_modes))
        ax.set_xticklabels([f'{idx}' for idx in sorted_indices])
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_reconstruction_error(self, figsize=(10, 6)):
        """
        Plot the reconstruction error as a function of time.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the error plot.
        """
        # Reconstruct the data
        reconstructed = self.reconstruct()
        
        # For real-valued data, take the real part
        if np.isreal(self.data).all():
            reconstructed = reconstructed.real
        
        # Calculate error for each time step
        errors = np.linalg.norm(self.data - reconstructed, axis=0) / np.linalg.norm(self.data, axis=0)
        times = np.arange(self.n_snapshots) * self.dt
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, errors, 'o-', markersize=4)
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Error')
        ax.set_title('DMD Reconstruction Error')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Calculate mean error
        mean_error = np.mean(errors)
        ax.axhline(mean_error, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean Error: {mean_error:.4f}')
        
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_time_dynamics(self, mode_indices=None, t_span=None, figsize=(12, 8)):
        """
        Plot the time dynamics of selected DMD modes.
        
        Parameters
        ----------
        mode_indices : list, optional
            Indices of modes to visualize. If None, visualizes the top modes by amplitude.
        t_span : tuple, optional
            Time span for visualization (t_start, t_end). If None, uses the original time span.
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the time dynamics plot.
        """
        if mode_indices is None:
            # Get the top modes by amplitude
            mode_indices = np.argsort(np.abs(self.alpha))[-4:][::-1]
        
        if t_span is None:
            t_span = (0, (self.n_snapshots - 1) * self.dt)
        
        # Create time vector
        times = np.linspace(t_span[0], t_span[1], 100)
        
        # Create time dynamics for selected modes
        time_dynamics = np.zeros((len(mode_indices), len(times)), dtype=complex)
        for j, mode_idx in enumerate(mode_indices):
            omega_j = self.omega[mode_idx]
            alpha_j = self.alpha[mode_idx]
            for i, t in enumerate(times):
                time_dynamics[j, i] = alpha_j * np.exp(omega_j * t)
        
        # Create figure
        fig, axes = plt.subplots(len(mode_indices), 2, figsize=figsize)
        
        if len(mode_indices) == 1:
            axes = axes.reshape(1, 2)
        
        for i, mode_idx in enumerate(mode_indices):
            # Real part
            ax = axes[i, 0]
            ax.plot(times, np.real(time_dynamics[i]), 'b-', label='Real Part')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Mode {mode_idx} - Real Part')
            ax.grid(True, alpha=0.3)
            
            # Imaginary part
            ax = axes[i, 1]
            ax.plot(times, np.imag(time_dynamics[i]), 'r-', label='Imaginary Part')
            ax.set_xlabel('Time')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Mode {mode_idx} - Imaginary Part')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_mode_contributions(self, n_modes=10, figsize=(10, 6)):
        """
        Plot the contribution of each mode to the overall dynamics.
        
        Parameters
        ----------
        n_modes : int, optional
            Number of modes to include in the plot.
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the contribution plot.
        """
        # Calculate mode contribution based on amplitude and eigenvalue magnitude
        amplitudes = np.abs(self.alpha)
        magnitudes = np.abs(self.Lambda)
        
        # For growing modes (|λ| > 1), scale by the growth over the time span
        final_time = (self.n_snapshots - 1) * self.dt
        for i, mag in enumerate(magnitudes):
            if mag > 1:
                magnitudes[i] = mag ** final_time
        
        # Calculate contribution as amplitude times magnitude
        contributions = amplitudes * magnitudes
        total_contribution = np.sum(contributions)
        normalized_contributions = contributions / total_contribution if total_contribution > 0 else contributions
        
        # Sort by contribution
        sorted_indices = np.argsort(normalized_contributions)[::-1]
        
        # Select top n_modes
        n_modes = min(n_modes, len(sorted_indices))
        top_indices = sorted_indices[:n_modes]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create pie chart
        labels = [f'Mode {idx}\n{normalized_contributions[idx]:.2%}' for idx in top_indices]
        wedges, texts = ax.pie(normalized_contributions[top_indices], 
                                           labels=None, 
                                           autopct=None,
                                           startangle=90,
                                           counterclock=False,
                                           wedgeprops={'edgecolor': 'w', 'linewidth': 1})
        
        # Create legend
        ax.legend(wedges, labels, title="Mode Contributions", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_title('DMD Mode Contributions to Overall Dynamics')
        
        plt.tight_layout()
        return fig
    
    def create_mode_animation(self, mode_idx, t_span=None, n_frames=50, grid_size=None, figsize=(10, 8)):
        """
        Create an animation of a DMD mode over time.
        
        Parameters
        ----------
        mode_idx : int
            Index of the mode to animate.
        t_span : tuple, optional
            Time span for animation (t_start, t_end). If None, uses the original time span.
        n_frames : int, optional
            Number of frames in the animation.
        grid_size : tuple, optional
            Grid size for reshaping the spatial modes (for 2D data).
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation of the mode over time.
        """
        if t_span is None:
            t_span = (0, (self.n_snapshots - 1) * self.dt)
        
        times = np.linspace(t_span[0], t_span[1], n_frames)
        
        # Get the mode
        mode = self.phi[:, mode_idx]
        omega = self.omega[mode_idx]
        alpha = self.alpha[mode_idx]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define update function for animation
        def update(frame):
            ax.clear()
            t = times[frame]
            
            # Calculate mode at time t
            mode_t = mode * alpha * np.exp(omega * t)
            
            # For real-valued data, take the real part
            if np.isreal(self.data).all():
                mode_t = np.real(mode_t)
            
            # Plot mode
            if grid_size is not None:
                im = ax.imshow(np.real(mode_t).reshape(grid_size), cmap='RdBu_r')
                ax.set_title(f'Mode {mode_idx} at t = {t:.2f}')
            else:
                ax.plot(np.real(mode_t))
                ax.set_ylim(np.min(np.real(mode)) * 1.5, np.max(np.real(mode)) * 1.5)
                ax.set_title(f'Mode {mode_idx} at t = {t:.2f}')
                ax.grid(True)
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
        plt.tight_layout()
        
        return anim
    
    def plot_3d_spectrum(self, figsize=(12, 10)):
        """
        Create a 3D plot of the DMD spectrum with eigenvalues, growth rates, and frequencies.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the 3D spectrum plot.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract properties
        growth_rates = np.real(self.omega)
        frequencies = np.imag(self.omega) / (2 * np.pi)  # Convert to cycles per time unit
        amplitudes = np.abs(self.alpha)
        eigenvalue_magnitudes = np.abs(self.Lambda)
        
        # Normalize amplitudes for point size
        normalized_amplitudes = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else amplitudes
        
        # Create scatter plot
        scatter = ax.scatter(growth_rates, frequencies, eigenvalue_magnitudes, 
                            s=normalized_amplitudes*200, 
                            c=eigenvalue_magnitudes, 
                            cmap='viridis', 
                            alpha=0.7)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Eigenvalue Magnitude |λ|')
        
        # Set labels and title
        ax.set_xlabel('Growth Rate')
        ax.set_ylabel('Frequency (cycles per time unit)')
        ax.set_zlabel('Eigenvalue Magnitude |λ|')
        ax.set_title('3D DMD Spectrum')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Annotate significant modes
        significant_modes = np.argsort(normalized_amplitudes)[-5:]
        for i in significant_modes:
            ax.text(growth_rates[i], frequencies[i], eigenvalue_magnitudes[i], f'{i}', fontsize=10)
        
        return fig
    
    def plot_svd_analysis(self, figsize=(12, 6)):
        """
        Plot the singular values and their cumulative energy to help with rank selection.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the SVD analysis plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot singular values
        singular_values = self.sigma
        ax1.semilogy(np.arange(1, len(singular_values) + 1), singular_values, 'o-', markersize=6)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value (log scale)')
        ax1.set_title('Singular Values')
        ax1.grid(True)
        
        # Plot cumulative energy
        energy = singular_values**2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        ax2.plot(np.arange(1, len(singular_values) + 1), cumulative_energy, 'o-', markersize=6)
        ax2.set_xlabel('Number of Modes')
        ax2.set_ylabel('Cumulative Energy')
        ax2.set_title('Cumulative Energy of Singular Values')
        
        # Add horizontal lines at common threshold levels
        thresholds = [0.9, 0.95, 0.99]
        for thresh in thresholds:
            ax2.axhline(thresh, color='r', linestyle='--', alpha=0.5)
            # Find index where cumulative energy exceeds threshold
            idx = np.argmax(cumulative_energy >= thresh)
            ax2.text(len(singular_values) * 0.7, thresh + 0.01, f'{thresh:.0%}: {idx+1} modes', 
                    verticalalignment='bottom')
        
        ax2.grid(True)
        plt.tight_layout()
        
        return fig
    
    def generate_diagnostic_report(self):
        """
        Generate a comprehensive diagnostic report of the DMD analysis.
        
        Returns
        -------
        str
            Report text containing diagnostic information.
        """
        # Calculate basic metrics
        residual_info = self.residual_analysis()
        spectral_radius = self.eigenvalue_check()
        
        # Create mode frequency table
        mode_freq_df = self.mode_frequencies().head(10)
        
        # Calculate system stability
        is_stable = spectral_radius <= 1.0
        
        # Identify dominant modes
        significant_modes = self.mode_significance().head(5)
        
        # Generate report text
        report = "# DMD Analysis Diagnostic Report\n\n"
        
        # Basic information
        report += "## System Overview\n"
        report += f"- Number of features: {self.n_features}\n"
        report += f"- Number of snapshots: {self.n_snapshots}\n"
        report += f"- Time step (dt): {self.dt}\n"
        report += f"- Rank used: {self.effective_rank}\n\n"
        
        # System stability
        report += "## System Stability\n"
        report += f"- Spectral radius: {spectral_radius:.4f}\n"
        report += f"- System stability: {'Stable' if is_stable else 'Unstable'}\n"
        if not is_stable:
            report += "  ⚠️ The system has growing modes which indicate instability or transient growth.\n"
        
        # Reconstruction accuracy
        report += "\n## Reconstruction Accuracy\n"
        report += f"- Relative error: {residual_info['relative_error']:.4e}\n"
        report += f"- Maximum absolute error: {residual_info['max_error']:.4e}\n"
        report += f"- Mean absolute error: {residual_info['mean_error']:.4e}\n"
        
        # Most significant modes
        report += "\n## Most Significant Modes\n"
        report += significant_modes.to_markdown() + "\n"
        
        # Mode frequencies
        report += "\n## Mode Frequency Analysis\n"
        report += mode_freq_df.to_markdown() + "\n"
        
        # Recommendations
        report += "\n## Recommendations\n"
        
        # Rank selection recommendation
        if self.rank is None:
            report += "- Consider using rank truncation to filter out noise and improve computational efficiency.\n"
        
        # Stability recommendations
        if not is_stable:
            report += "- Investigate growing modes to understand system instabilities.\n"
            report += "- For prediction tasks, be cautious about extrapolating too far into the future.\n"
        
        # Mode selection recommendations
        report += f"- Focus analysis on the top {min(5, len(significant_modes))} modes by significance for key dynamics.\n"
        
        # Model interpretation
        report += "\n## Physical Interpretation\n"
        report += "- DMD modes represent coherent structures in the data with specific frequencies and growth/decay rates.\n"
        report += "- Modes with eigenvalues close to the unit circle represent persistent dynamics.\n"
        report += "- Modes with eigenvalues inside the unit circle represent decaying dynamics.\n"
        report += "- Modes with eigenvalues outside the unit circle represent growing dynamics.\n"
        
        return report
    
    def forecast(self, future_times):
        """
        Forecast future states of the system.
        
        Parameters
        ----------
        future_times : array-like
            Times at which to forecast the state of the system.
            
        Returns
        -------
        numpy.ndarray
            Forecasted states at specified times.
        dict
            Dictionary containing confidence information.
        """
        # Forecast using DMD
        forecasted_states = self.predict(future_times)
        
        # Calculate confidence based on extrapolation distance and system stability
        spectral_radius = np.max(np.abs(self.Lambda))
        max_observed_time = (self.n_snapshots - 1) * self.dt
        confidence = {}
        
        for i, t in enumerate(future_times):
            # For times within observed range, confidence is high
            if t <= max_observed_time:
                confidence_value = 0.9
            else:
                # Decrease confidence based on extrapolation distance and stability
                extrapolation_factor = (t - max_observed_time) / max_observed_time
                
                # For stable systems, confidence decreases more slowly
                if spectral_radius <= 1.0:
                    confidence_value = 0.9 * np.exp(-0.5 * extrapolation_factor)
                else:
                    confidence_value = 0.9 * np.exp(-2.0 * extrapolation_factor * spectral_radius)
                
                confidence_value = max(0.1, confidence_value)  # Set minimum confidence
            
            confidence[t] = confidence_value
        
        return forecasted_states, confidence
    
    def compute_koopman_modes(self):
        """
        Compute the Koopman modes which are the left eigenvectors of the DMD operator
        using an SVD-based approach to avoid constructing the full DMD operator.
        
        Returns
        -------
        numpy.ndarray
            Koopman modes.
        """

        # Use SVD of X_1 to calculate pseudoinverse more efficiently
        U, s, Vh = np.linalg.svd(self.X_1, full_matrices=False)
        
        r = len(s)
        U_r = U[:, :r]
        s_r = s[:r]
        Vh_r = Vh[:r, :]
        
        # Compute the reduced operator
        S_inv = np.diag(1.0 / s_r)

        A_tilde = U_r.T @ self.X_2 @ Vh_r.T @ S_inv
        
        # Compute eigendecomposition of the reduced operator's transpose
        eigenvalues, eigenvectors_tilde = np.linalg.eig(A_tilde.T)
        
        # Transform reduced eigenvectors to Koopman modes
        # Left eigenvectors of A are V * S_inv * U.T * eigenvectors
        koopman_modes = eigenvectors_tilde @ Vh_r 
        
        return koopman_modes
    
    def compute_dmd_for_time(self, t):
        """
        Compute the state of the system at a specific time using DMD.
        
        Parameters
        ----------
        t : float
            Time at which to compute the state.
            
        Returns
        -------
        numpy.ndarray
            State of the system at time t.
        """
        # Compute time dynamics
        time_dynamics = np.exp(self.omega * t)
        
        # Compute state
        state = self.phi @ (self.alpha * time_dynamics)
        
        # For real-valued data, take the real part
        if np.isreal(self.data).all():
            state = state.real
            
        return state
    
    def plot_eigenfunction_evolution(self, eigenfunction_idx, t_span=None, n_points=100, figsize=(10, 6)):
        """
        Plot the evolution of a Koopman eigenfunction over time.
        
        Parameters
        ----------
        eigenfunction_idx : int
            Index of the eigenfunction to visualize.
        t_span : tuple, optional
            Time span for visualization (t_start, t_end). If None, uses the original time span.
        n_points : int, optional
            Number of points in the time grid.
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the eigenfunction evolution plot.
        """
        if t_span is None:
            t_span = (0, (self.n_snapshots - 1) * self.dt)
        
        # Create time vector
        times = np.linspace(t_span[0], t_span[1], n_points)
        
        # Get eigenvalue and initial condition
        lambda_k = self.Lambda[eigenfunction_idx]
        omega_k = self.omega[eigenfunction_idx]
        
        # Compute eigenfunction values
        eigenfunction_values = np.zeros(len(times), dtype=complex)
        for i, t in enumerate(times):
            eigenfunction_values[i] = np.exp(omega_k * t)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot real part
        ax1.plot(times, np.real(eigenfunction_values), 'b-')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Real Part')
        ax1.set_title(f'Eigenfunction {eigenfunction_idx} (Real Part)')
        ax1.grid(True)
        
        # Plot imaginary part
        ax2.plot(times, np.imag(eigenfunction_values), 'r-')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Imaginary Part')
        ax2.set_title(f'Eigenfunction {eigenfunction_idx} (Imaginary Part)')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def explain_dmd_modes(self, n_modes=5):
        """
        Provide a text explanation of the top DMD modes.
        
        Parameters
        ----------
        n_modes : int, optional
            Number of modes to explain.
            
        Returns
        -------
        str
            Text explanation of the modes.
        """
        # Get mode information
        mode_df = self.mode_frequencies()
        significant_modes = self.mode_significance().head(n_modes)
        
        # Generate explanation text
        explanation = "# DMD Mode Explanation\n\n"
        
        for _, row in significant_modes.iterrows():
            mode_idx = int(row['Mode'])
            freq = mode_df.loc[mode_df['Mode'] == mode_idx, 'Frequency'].values[0]
            growth_rate = mode_df.loc[mode_df['Mode'] == mode_idx, 'Growth_Rate'].values[0]
            magnitude = row['Magnitude']
            amplitude = row['Normalized_Amplitude']
            
            explanation += f"## Mode {mode_idx}\n"
            explanation += f"- Significance: {row['Significance']:.4f}\n"
            explanation += f"- Normalized Amplitude: {amplitude:.4f}\n"
            explanation += f"- Frequency: {freq:.4f} cycles per time unit\n"
            explanation += f"- Growth Rate: {growth_rate:.4f}\n"
            explanation += f"- Eigenvalue Magnitude: {magnitude:.4f}\n\n"
            
            # Add interpretation
            explanation += "### Interpretation:\n"
            
            # Frequency interpretation
            if abs(freq) < 1e-5:
                explanation += "- This is a non-oscillatory mode (zero frequency).\n"
            elif abs(freq) < 0.1:
                explanation += "- This mode has a low frequency, representing slow-varying dynamics.\n"
            else:
                explanation += f"- This mode oscillates at a frequency of {freq:.4f} cycles per time unit.\n"
            
            # Growth rate interpretation
            if abs(growth_rate) < 1e-5:
                explanation += "- This mode neither grows nor decays (marginal stability).\n"
            elif growth_rate < 0:
                explanation += f"- This is a decaying mode with decay rate {abs(growth_rate):.4f}.\n"
                explanation += f"- It will decrease by a factor of {np.exp(growth_rate * 10):.4f} after 10 time units.\n"
            else:
                explanation += f"- This is a growing mode with growth rate {growth_rate:.4f}.\n"
                explanation += f"- It will increase by a factor of {np.exp(growth_rate * 10):.4f} after 10 time units.\n"
            
            # Significance interpretation
            if row['Significance'] > 0.5:
                explanation += "- This is a dominant mode that significantly influences the system dynamics.\n"
            elif row['Significance'] > 0.1:
                explanation += "- This mode has moderate influence on the system dynamics.\n"
            else:
                explanation += "- This mode has minor influence on the system dynamics.\n"
            
            explanation += "\n"
        
        return explanation
    
    def set_adaptive_dt(self, reference_time):
        """
        Set the adaptive time step using a reference time unit.
        
        Parameters
        ----------
        reference_time : float
            Reference time unit.
            
        Returns
        -------
        float
            Newly set dt.
        """
        self.dt = reference_time / self.n_snapshots
        
        # Recompute omega (continuous-time eigenvalues)
        self.omega = np.log(self.Lambda) / self.dt
        
        return self.dt
    
    def compute_optimal_prediction_rank(self, test_ratio=0.2, max_rank=None):
        """
        Compute the optimal rank for prediction by cross-validation.
        
        Parameters
        ----------
        test_ratio : float, optional
            Ratio of snapshots to use for testing.
        max_rank : int, optional
            Maximum rank to consider. If None, uses min(n_features, n_snapshots-1).
            
        Returns
        -------
        int
            Optimal rank for prediction.
        dict
            Dictionary containing error information for different ranks.
        """
        # Determine maximum rank to test
        if max_rank is None:
            max_rank = min(self.n_features, self.n_snapshots - 1)
        
        # Split data into training and testing
        n_test = int(test_ratio * self.n_snapshots)
        n_train = self.n_snapshots - n_test
        
        train_data = self.data[:, :n_train]
        test_data = self.data[:, n_train:]
        
        # Test different ranks
        ranks_to_test = range(1, max_rank + 1)
        errors = {}
        
        for r in ranks_to_test:
            # Create DMD model with current rank
            dmd_model = DMD(train_data, dt=self.dt, rank=r)
            
            # Predict test times
            test_times = np.arange(n_train, self.n_snapshots) * self.dt
            predicted = dmd_model.predict(test_times)
            
            # For real-valued data, take the real part
            if np.isreal(test_data).all():
                predicted = predicted.real
            
            # Calculate error
            error = np.linalg.norm(test_data - predicted) / np.linalg.norm(test_data)
            errors[r] = error
        
        # Find optimal rank
        optimal_rank = min(errors, key=errors.get)
        
        return optimal_rank, errors
    
    def is_mode_physical(self, mode_idx, threshold=0.1):
        """
        Determine if a mode is likely to be physical rather than noise.
        
        Parameters
        ----------
        mode_idx : int
            Index of the mode to check.
        threshold : float, optional
            Significance threshold.
            
        Returns
        -------
        bool
            True if the mode is likely physical, False otherwise.
        """
        # Get mode significance
        sig_df = self.mode_significance()
        mode_sig = sig_df.loc[sig_df['Mode'] == mode_idx, 'Significance'].values[0]
        
        # Check if the mode is significant
        is_significant = mode_sig > threshold
        
        # Check if the mode is well separated from other eigenvalues
        eigenvalues = self.Lambda
        current_eigenvalue = eigenvalues[mode_idx]
        
        distances = np.abs(eigenvalues - current_eigenvalue)
        distances[mode_idx] = float('inf')  # Exclude self
        min_distance = np.min(distances)
        
        is_well_separated = min_distance > 0.05
        
        # Check if the mode is coherent (smooth in space)
        mode = self.phi[:, mode_idx]
        mode_normalized = mode / np.max(np.abs(mode))
        
        # Calculate a simple smoothness measure (can be improved for specific applications)
        diffs = np.diff(np.abs(mode_normalized))
        smoothness = 1.0 / (1.0 + np.mean(np.abs(diffs)))
        
        is_smooth = smoothness > 0.7
        
        # Combine criteria
        is_physical = is_significant and (is_well_separated or is_smooth)
        
        return is_physical
    
    def plot_complex_alpha(self, figsize=(8, 8), annotate=False):
        """
        Plot the complex mode amplitudes in the complex plane.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size.
        annotate : bool, optional
            Whether to annotate the points with mode indices.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the complex plane plot.
        """
        z = self.alpha
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        scatter = ax.scatter(z.real, z.imag, c=np.abs(z), cmap='viridis', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Amplitude |α|')
        
        # Annotate points if requested
        if annotate:
            for i, val in enumerate(z):
                ax.text(z[i].real + 0.1, z[i].imag, f'{i}', fontsize=9)
        
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title('Mode Amplitudes in Complex Plane')
        ax.grid(True)
        ax.set_aspect('equal')
        
        return fig
    
    def advect_state(self, state, time_delta):
        """
        Advect a state forward in time using the DMD model.
        
        Parameters
        ----------
        state : numpy.ndarray
            Initial state vector.
        time_delta : float
            Time to advect forward.
            
        Returns
        -------
        numpy.ndarray
            Advected state.
        """
        # Project state onto DMD modes
        mode_amplitudes = np.linalg.pinv(self.phi) @ state
        
        # Evolve in time
        evolved_amplitudes = mode_amplitudes * np.exp(self.omega * time_delta)
        
        # Project back to state space
        advected_state = self.phi @ evolved_amplitudes
        
        # For real-valued data, take the real part
        if np.isreal(self.data).all():
            advected_state = advected_state.real
        
        return advected_state
    
    def compute_exact_dmd(self):
        """
        Compute the exact DMD according to Tu et al. (2014).
        
        Returns
        -------
        numpy.ndarray
            Exact DMD modes.
        """
        # SVD of X_1
        U, sigma, Vh = np.linalg.svd(self.X_1, full_matrices=False)
        V = Vh.T
        
        # Truncate if rank is specified
        if self.rank is not None:
            r = min(self.rank, len(sigma))
            U = U[:, :r]
            sigma = sigma[:r]
            V = V[:, :r]
        
        # Create Sigma inverse
        Sigma_inv = np.diag(1.0 / sigma)
        
        # Compute A tilde (low-rank approximation of A)
        A_tilde = U.T @ self.X_2 @ V @ Sigma_inv
        
        # Eigendecomposition of A_tilde
        Lambda, W = np.linalg.eig(A_tilde)
        
        # Compute exact DMD modes
        exact_modes = self.X_2 @ V @ Sigma_inv @ W
        
        return exact_modes
    
    def compute_companion_dmd(self):
        """
        Compute the companion-form DMD, useful for very high-dimensional systems.
        
        Returns
        -------
        numpy.ndarray
            Companion matrix.
        numpy.ndarray
            Companion DMD modes.
        """
        # Create companion matrix
        n = self.n_snapshots - 1
        companion = np.zeros((n, n))
        
        # Fill the subdiagonal with ones
        for i in range(n-1):
            companion[i+1, i] = 1
        
        # Compute the last row
        last_row = np.linalg.lstsq(self.X_1.T, self.X_2[:, -1], rcond=None)[0]
        companion[0, :] = last_row
        
        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(companion)
        
        # Compute DMD modes
        modes = self.X_1 @ eigenvectors
        
        return companion, modes
    
    def plot_mode_phase_portrait(self, mode_idx1, mode_idx2, figsize=(10, 8)):
        """
        Plot a phase portrait of two DMD modes.
        
        Parameters
        ----------
        mode_idx1 : int
            Index of the first mode.
        mode_idx2 : int
            Index of the second mode.
        figsize : tuple, optional
            Figure size.
            
        Returns
        -------
        matplotlib.figure.Figure
            The figure object containing the phase portrait.
        """
        # Get mode dynamics
        times = np.arange(self.n_snapshots) * self.dt
        
        mode1_dynamics = np.zeros(len(times), dtype=complex)
        mode2_dynamics = np.zeros(len(times), dtype=complex)
        
        for i, t in enumerate(times):
            mode1_dynamics[i] = self.alpha[mode_idx1] * np.exp(self.omega[mode_idx1] * t)
            mode2_dynamics[i] = self.alpha[mode_idx2] * np.exp(self.omega[mode_idx2] * t)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot phase portrait
        ax.plot(np.real(mode1_dynamics), np.real(mode2_dynamics), 'b-')
        ax.scatter(np.real(mode1_dynamics[0]), np.real(mode2_dynamics[0]), color='green', s=100, label='Start')
        ax.scatter(np.real(mode1_dynamics[-1]), np.real(mode2_dynamics[-1]), color='red', s=100, label='End')
        
        # Set labels and title
        ax.set_xlabel(f'Mode {mode_idx1} (Real Part)')
        ax.set_ylabel(f'Mode {mode_idx2} (Real Part)')
        ax.set_title(f'Phase Portrait of Modes {mode_idx1} and {mode_idx2}')
        
        # Add grid and legend
        ax.grid(True)
        ax.legend()
        
        return fig
    
    def __str__(self):
        """String representation of the DMD object."""
        return (f"DMD(n_features={self.n_features}, n_snapshots={self.n_snapshots}, "
                f"dt={self.dt}, rank={self.rank})")
    
    def __repr__(self):
        """Representation of the DMD object."""
        return self.__str__()