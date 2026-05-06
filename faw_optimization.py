import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 1. THE MATHEMATICAL MODEL (ODE System)
def faw_model(y, t, T):
    E, L, A = y  # Eggs, Larvae, Adults
    
    # Temperature-dependent development rate (Simulating thermal biology)
    # FAW develops faster at higher temperatures (e.g., 28°C vs 20°C)
    growth_rate = 0.01 * (T - 10) if T > 10 else 0
    
    # Parameters
    oviposition_rate = 15  # Eggs laid per adult
    natural_death = 0.05
    
    # OPTIMAL CONTROL VARIABLE (u)
    # u = 1 means 100% spray effectiveness. We apply it when Larvae peak.
    u = 0.9 if (t > 15 and t < 25) else 0.0 
    
    # Differential Equations
    dE_dt = (oviposition_rate * A) - (growth_rate * E) - (natural_death * E)
    dL_dt = (growth_rate * E) - (growth_rate * L) - (natural_death * L) - (u * L)
    dA_dt = (growth_rate * L) - (natural_death * A)
    
    return [dE_dt, dL_dt, dA_dt]

# 2. SIMULATION SETTINGS
time = np.linspace(0, 50, 500) # 50 days
initial_pop = [100, 0, 5]     # 100 Eggs, 0 Larvae, 5 Adults
avg_temp = 27                  # Temperature in Celsius

# 3. SOLVE
solution = odeint(faw_model, initial_pop, time, args=(avg_temp,))
E, L, A = solution.T

# 4. VISUALIZATION
plt.figure(figsize=(10, 6))
plt.plot(time, E, label='Eggs', color='orange')
plt.plot(time, L, label='Larvae (Crop Damage Stage)', color='red', linewidth=2)
plt.plot(time, A, label='Adults', color='green')
plt.fill_between(time, 0, 2000, where=(time > 15) & (time < 25), 
                 color='grey', alpha=0.2, label='Optimal Control Window')
plt.title(f'FAW Population Dynamics at {avg_temp}°C')
plt.xlabel('Days')
plt.ylabel('Population Count')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
