# Dynamic Mode Decomposition (DMD) Toolkit

A comprehensive Python toolkit for Dynamic Mode Decomposition (DMD) analysis of dynamical systems. This toolkit provides powerful tools for modal analysis, feature extraction, and prediction of complex time-dependent systems.

## Overview

Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing complex dynamical systems. It extracts spatiotemporal coherent structures (modes) from high-dimensional data, allowing for:

- System identification and modal analysis
- Low-rank approximation of dynamics
- Feature extraction and pattern recognition
- Prediction and forecasting
- Koopman operator approximation

This toolkit implements the DMD algorithm and provides extensive tools for visualization, analysis, and diagnostics.

## Features

- **Core DMD Implementation**:
  - Standard and exact DMD computation
  - SVD-based dimensionality reduction with optional rank truncation
  - Eigenvalue decomposition to identify modes, frequencies, and growth rates

- **Mode Analysis**:
  - Mode significance ranking
  - Frequency and growth rate calculations
  - Physical mode detection
  - Koopman mode computation

- **Visualization Tools**:
  - DMD spectrum plots (discrete and continuous-time)
  - Mode amplitude and frequency plots
  - Spatial mode visualization
  - 3D spectrum plots
  - SVD analysis for rank selection
  - Complex plane visualizations
  - Time dynamics animations
  - Phase portraits

- **Diagnostics**:
  - Reconstruction error analysis
  - Eigenvalue stability checking
  - Mode orthogonality verification
  - Residual analysis

- **Prediction Capabilities**:
  - Future state prediction
  - Confidence estimation for forecasts
  - Optimal rank selection via cross-validation
  - State advection in time

- **Specialized Application Examples**:
  - Lorenz system analysis
  - Generic data processing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dmd-toolkit.git
cd dmd-toolkit

# Install required packages
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import numpy as np
from dmd import DMD

# Create or load data matrix (each column is a snapshot, each row is a feature)
data = np.random.rand(100, 50)  # 100 features, 50 time snapshots
dt = 0.1  # time step between snapshots

# Create DMD object
dmd = DMD(data, dt=dt)

# Analyze modes
frequencies = dmd.mode_frequencies()
significance = dmd.mode_significance()

# Plot spectrum
dmd.plot_spectrum()

# Make predictions
future_times = np.linspace(0, 10, 100)
predictions = dmd.predict(future_times)
```

### Using the DMD Plotter

The toolkit includes a specialized `DMDPlotter` class for creating high-quality visualizations:

```python
from dmd import DMD
from dmd_plotting import DMDPlotter

# Initialize DMD model
dmd = DMD(data, dt=0.1)

# Initialize plotter
plotter = DMDPlotter(dmd)

# Create visualizations
plotter.plot_spectrum(figsize=(12, 8), save_path="spectrum.png")
plotter.plot_mode_amplitudes(n_modes=10, save_path="amplitudes.png")
plotter.spatial_mode_visualization(mode_indices=[0, 1, 2, 3], save_path="modes.png")
```

### Example: Lorenz System Analysis

The toolkit includes example code for analyzing the Lorenz system:

```python
from lorenz_test import main as lorenz_main

# Run the Lorenz system DMD analysis
dmd = lorenz_main()
```

### General Analysis Script

For analyzing your own data:

```python
import numpy as np
from test import main

# Load your data
data = np.load("your_data.npy")

# Run DMD analysis
dmd = main(data)
```

## File Structure

- `dmd.py`: Core DMD class implementation
- `dmd_plotting.py`: Enhanced plotting functionality for DMD analysis
- `test.py`: General test script for arbitrary data
- `lorenz_test.py`: Specialized example for Lorenz system analysis

## Theory

Dynamic Mode Decomposition approximates the dynamics of a system as:

x(t+1) ≈ A x(t)

where A is the linear operator that best maps the data from one time step to the next. DMD computes the eigendecomposition of A to extract:

- Eigenvalues (λ): Represent growth/decay rates and oscillation frequencies
- Eigenvectors (ϕ): Represent spatial structures (modes)

The time evolution of each mode follows e^(ωt) where ω = log(λ)/dt.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{dmd_toolkit,
  author = {Ameir Shaa, Claude Guet},
  <!-- title = {Dynamic Mode Decomposition Toolkit}, -->
  <!-- year = {2025}, -->
  <!-- url = {https://github.com/yourusername/dmd-toolkit} -->
}
```

## References

1. Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of Fluid Mechanics, 656, 5-28.
2. Tu, J. H., Rowley, C. W., Luchtenburg, D. M., Brunton, S. L., & Kutz, J. N. (2014). On dynamic mode decomposition: Theory and applications. Journal of Computational Dynamics, 1(2), 391-421.
3. Kutz, J. N., Brunton, S. L., Brunton, B. W., & Proctor, J. L. (2016). Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM.