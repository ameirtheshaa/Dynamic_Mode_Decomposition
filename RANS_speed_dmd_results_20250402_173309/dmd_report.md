# DMD Analysis Diagnostic Report

## System Overview
- Number of features: 352071
- Number of snapshots: 24
- Time step (dt): 15
- Rank used: 23

## System Stability
- Spectral radius: 1.0005
- System stability: Unstable
  ⚠️ The system has growing modes which indicate instability or transient growth.

## Reconstruction Accuracy
- Relative error: 2.5684e-02
- Maximum absolute error: 3.8961e-01
- Mean absolute error: 8.2264e-03

## Most Significant Modes
|    |   Mode |   Significance |   Normalized_Amplitude |   Magnitude |
|---:|-------:|---------------:|-----------------------:|------------:|
|  0 |      0 |      1.00051   |              1         |    1.00051  |
|  2 |      2 |      0.0531566 |              0.053187  |    0.999429 |
|  1 |      1 |      0.0531566 |              0.053187  |    0.999429 |
|  4 |      4 |      0.0468851 |              0.0470182 |    0.997169 |
|  3 |      3 |      0.0468851 |              0.0470182 |    0.997169 |

## Mode Frequency Analysis
|    |   Mode |   Frequency |   Growth_Rate |   Magnitude |   Amplitude |   Normalized_Amplitude |
|---:|-------:|------------:|--------------:|------------:|------------:|-----------------------:|
|  0 |      0 |  0          |   3.41563e-05 |    1.00051  |   424.721   |              0.724865  |
|  2 |      2 | -0.00275627 |  -3.80635e-05 |    0.999429 |    22.5896  |              0.0385534 |
|  1 |      1 |  0.00275627 |  -3.80635e-05 |    0.999429 |    22.5896  |              0.0385534 |
|  4 |      4 | -0.00557868 |  -0.000188989 |    0.997169 |    19.9696  |              0.0340819 |
|  3 |      3 |  0.00557868 |  -0.000188989 |    0.997169 |    19.9696  |              0.0340819 |
|  6 |      6 | -0.0083773  |  -0.000876226 |    0.986943 |     9.26112 |              0.0158058 |
|  5 |      5 |  0.0083773  |  -0.000876226 |    0.986943 |     9.26112 |              0.0158058 |
|  7 |      7 |  0.0112517  |  -0.00187739  |    0.972232 |     6.43371 |              0.0109803 |
|  8 |      8 | -0.0112517  |  -0.00187739  |    0.972232 |     6.43371 |              0.0109803 |
|  9 |      9 |  0.014071   |  -0.00223468  |    0.967035 |     4.63806 |              0.0079157 |

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
