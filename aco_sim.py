import numpy as np
import matplotlib.pyplot as plt

def american_call_dividend_binomial(S, K, T, r, q, sigma, N):
    dt = T / N
    df = np.exp(-r * dt)
    growth_factor = np.exp((r - q) * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (growth_factor - d) / (u - d)

    # Stock price tree initialization
    stock_price = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_price[j, i] = S * (u ** (i - j)) * (d ** j)

    # Option value tree
    option_value = np.zeros_like(stock_price)
    option_value[:, N] = np.maximum(stock_price[:, N] - K, 0)


    # Backward induction
    # compute exercise boundary
    exercise_boundary = np.zeros(N)
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            hold = (p * option_value[j, i + 1] + (1 - p) * option_value[j + 1, i + 1]) * df
            exercise = stock_price[j, i] - K
            option_value[j, i] = max(hold, exercise)
            
            # if we have an exercise greater than hold, we set it as the boundary
            if exercise > hold:
                exercise_boundary[i] = stock_price[j, i]
        

    return stock_price, option_value, exercise_boundary


# Parameters
S = 100
K = 100
T = 1
r = 0.05
q = 0.025
sigma = 0.20
N = 1000

stock_price, option_value, boundary = american_call_dividend_binomial(S, K, T, r, q, sigma, N)

# Time to expiry
tau = np.linspace(T, 0, N)
fig = plt.figure(figsize=(10, 6))
plt.plot(tau, boundary)
plt.title('Optimal Exercise Boundary of a Dividend-Paying American Call Option')
plt.xlabel('Time to Expiry (τ)')
plt.ylabel('Stock Price at Boundary')
plt.grid(True)
#plt.show()
fig.savefig('images/r_d.png')
