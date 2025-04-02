# DMD Analysis Diagnostic Report

## System Overview
- Number of features: 100
- Number of snapshots: 50
- Time step (dt): 15
- Rank used: 49

## System Stability
- Spectral radius: 1.0313
- System stability: Unstable
  ⚠️ The system has growing modes which indicate instability or transient growth.

## Reconstruction Accuracy
- Relative error: 9.6001e-02
- Maximum absolute error: 6.4346e-01
- Mean absolute error: 1.2620e-01

## Most Significant Modes
|    |   Mode |   Significance |   Normalized_Amplitude |   Magnitude |
|---:|-------:|---------------:|-----------------------:|------------:|
|  0 |      0 |       0.999993 |               1        |    0.999993 |
|  1 |      1 |       0.999993 |               1        |    0.999993 |
|  2 |      2 |       0.503267 |               0.503841 |    0.998859 |
|  3 |      3 |       0.503267 |               0.503841 |    0.998859 |
|  4 |      4 |       0.258177 |               0.25845  |    0.998942 |

## Mode Frequency Analysis
|    |   Mode |   Frequency |   Growth_Rate |   Magnitude |   Amplitude |   Normalized_Amplitude |
|---:|-------:|------------:|--------------:|------------:|------------:|-----------------------:|
|  0 |      0 |  0.00109068 |  -4.49656e-07 |    0.999993 |    9.84415  |              0.220624  |
|  1 |      1 | -0.00109068 |  -4.49656e-07 |    0.999993 |    9.84415  |              0.220624  |
|  2 |      2 |  0.00214413 |  -7.60816e-05 |    0.998859 |    4.95989  |              0.11116   |
|  3 |      3 | -0.00214413 |  -7.60816e-05 |    0.998859 |    4.95989  |              0.11116   |
|  4 |      4 |  0.00323811 |  -7.05536e-05 |    0.998942 |    2.54423  |              0.0570205 |
|  5 |      5 | -0.00323811 |  -7.05536e-05 |    0.998942 |    2.54423  |              0.0570205 |
| 28 |     28 |  0          |  -0.0355714   |    0.586507 |    0.961952 |              0.021559  |
| 18 |     18 |  0.0298118  |  -0.0023751   |    0.965001 |    0.50323  |              0.0112782 |
| 19 |     19 | -0.0298118  |  -0.0023751   |    0.965001 |    0.50323  |              0.0112782 |
| 45 |     45 |  0.0182402  |  -0.00485152  |    0.929812 |    0.468364 |              0.0104969 |

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
