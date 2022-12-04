# Based on 100 random distributions, analyze statistics of UI(X; Y \ Z) vs UI(Y; X \ Z)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

main_path = 'C:\\Users\\amand\\Documents\\research\\Project_Causal_PID\\ui_asymmetry.csv'
ui_arr = pd.read_csv(main_path)
ui = ui_arr['CMI']
kl = ui_arr['KL']
mi = ui_arr['MI']

# Correlation between difference in UI and MI of free variable
ui_diff = [ui[i+1]-ui[i] for i in np.arange(0, len(ui), 2)]
mi_diff = [mi[i+1]-mi[i] for i in np.arange(0, len(mi), 2)]
kl_diff = [kl[i+1]-kl[i] for i in np.arange(0, len(kl), 2)]

# Plot
plt.scatter(mi_diff, ui_diff, marker='.')
max_y = max([abs(u) for u in ui_diff])
max_x = max([abs(m) for m in mi_diff])
plt.plot([0, 0], [max_y, -max_y], 'k--')
plt.plot([max_x, -max_x], [0, 0], 'k--')

plt.xlabel("I(Y; Z) - I(X; Z)")
plt.ylabel("UI(X; Y \\ Z) - UI(Y; X \\ Z)")
plt.show()

print("Yo")

