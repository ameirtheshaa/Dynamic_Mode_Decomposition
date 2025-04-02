# DMD Analysis Diagnostic Report

## System Overview
- Number of features: 1000
- Number of snapshots: 100
- Time step (dt): 0.5050505050505051
- Rank used: 99

## System Stability
- Spectral radius: 1.1748
- System stability: Unstable
  ⚠️ The system has growing modes which indicate instability or transient growth.

## Reconstruction Accuracy
- Relative error: 1.9348e+07
- Maximum absolute error: 2.3850e+09
- Mean absolute error: 3.2855e+07

## Most Significant Modes
|    |   Mode |   Significance |   Normalized_Amplitude |   Magnitude |
|---:|-------:|---------------:|-----------------------:|------------:|
| 65 |     65 |        1.14625 |               1        |     1.14625 |
| 66 |     66 |        1.14625 |               1        |     1.14625 |
| 61 |     61 |        1.09892 |               0.955873 |     1.14965 |
| 62 |     62 |        1.09892 |               0.955873 |     1.14965 |
| 69 |     69 |        1.09398 |               0.959332 |     1.14035 |

## Mode Frequency Analysis
|    |   Mode |   Frequency |   Growth_Rate |   Magnitude |   Amplitude |   Normalized_Amplitude |
|---:|-------:|------------:|--------------:|------------:|------------:|-----------------------:|
| 65 |     65 |    0.552528 |      0.270266 |     1.14625 |     747.461 |              0.0235864 |
| 66 |     66 |   -0.552528 |      0.270266 |     1.14625 |     747.461 |              0.0235864 |
| 69 |     69 |    0.528526 |      0.260046 |     1.14035 |     717.063 |              0.0226272 |
| 70 |     70 |   -0.528526 |      0.260046 |     1.14035 |     717.063 |              0.0226272 |
| 61 |     61 |    0.576136 |      0.276134 |     1.14965 |     714.477 |              0.0225456 |
| 62 |     62 |   -0.576136 |      0.276134 |     1.14965 |     714.477 |              0.0225456 |
| 14 |     14 |    0.177909 |      0.318725 |     1.17465 |     690.772 |              0.0217976 |
| 15 |     15 |   -0.177909 |      0.318725 |     1.17465 |     690.772 |              0.0217976 |
| 10 |     10 |    0.154208 |      0.318903 |     1.17476 |     688.809 |              0.0217356 |
| 11 |     11 |   -0.154208 |      0.318903 |     1.17476 |     688.809 |              0.0217356 |

## Recommendations
- Consider using rank truncation to filter out noise and improve computational efficiency.
- Investigate growing modes to understand system instabilities.
- For prediction tasks, be cautious about extrapolating too far into the future.
- Focus analysis on the top 5 modes by significance for key dynamics.

## Physical Interpretation
- DMD modes represent coherent structures in the data with specific frequencies and growth/decay rates.
- Modes with eigenvalues close to the unit circle represent persistent dynamics.
- Modes with eigenvalues inside the unit circle represent decaying dynamics.
- Modes with eigenvalues outside the unit circle represent growing dynamics.
