# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# REGNO: 212223240082
# Date: 11-05-2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load IPL dataset
data = pd.read_csv('matches.csv')

# Convert date column
data['date'] = pd.to_datetime(data['date'], format='mixed', dayfirst=True)

# Extract year
data['Year'] = data['date'].dt.year

# Count matches per year
yearly_matches = data.groupby('Year')['id'].count().reset_index()

# Time series data
X = yearly_matches['id']

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Original Data Plot
plt.plot(yearly_matches['Year'], X, marker='o')
plt.title('Original IPL Matches Data')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.show()

# ACF and PACF
plt.subplot(2, 1, 1)
plot_acf(X, lags=min(5, len(X)//2), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=min(5, len(X)//2), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1) Model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot Simulated ARMA(1,1)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of Simulated ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of Simulated ARMA(1,1)')
plt.show()

# ARMA(2,2) Model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

# Plot Simulated ARMA(2,2)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of Simulated ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of Simulated ARMA(2,2)')
plt.show()
```

### OUTPUT:
Original Data:
<img width="1021" height="497" alt="image" src="https://github.com/user-attachments/assets/b3f9dcf2-7b11-42ba-a6d2-bfcbf184b790" />

Autocorrelation:
<img width="1175" height="267" alt="image" src="https://github.com/user-attachments/assets/d545b657-97bf-41bc-b8ea-4c1636aeba69" />

Partial Autocorrelation:
<img width="1161" height="271" alt="image" src="https://github.com/user-attachments/assets/d08a4df8-6325-4f60-820b-e43a0bfbfb17" />

SIMULATED ARMA(1,1) PROCESS:
<img width="977" height="495" alt="image" src="https://github.com/user-attachments/assets/922f7b5c-a04f-4aec-a1a4-d694f06dcc65" />

Partial Autocorrelation:
<img width="1050" height="477" alt="image" src="https://github.com/user-attachments/assets/8079c552-cca8-4c90-92ce-ca1ad535c377" />

Autocorrelation:
<img width="997" height="491" alt="image" src="https://github.com/user-attachments/assets/e135333c-c6ee-465c-8fce-3c7f8e5d7cd8" />

SIMULATED ARMA(2,2) PROCESS:
<img width="1001" height="487" alt="image" src="https://github.com/user-attachments/assets/9f198c52-38e2-440b-8839-79dfce8b8e3d" />

Autocorrelation:
<img width="987" height="491" alt="image" src="https://github.com/user-attachments/assets/d4fb6b65-9c90-4f98-a79c-383a8cc42eaa" />

Partial Autocorrelation:
<img width="992" height="488" alt="image" src="https://github.com/user-attachments/assets/ad7eef11-a4c7-4f7d-a692-4a2fd2e16daa" />
### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
