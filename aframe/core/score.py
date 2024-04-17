import numpy as np

# wing_loading = 15 # kg/m^2
# aspect_ratio = AR = 6
# gross_weight = 2 # kg
# payload_weight_fraction = f = 0.5
# length_width_ratio = LB = 0.6
# B = 1

# S = gross_weight / wing_loading # m^2

# flight_score = 20 * f * gross_weight + f - ((AR * S)**2 * (1 + LB)**4) + 20 * B
# print(flight_score) # 36.3

wing_loading = 71 # N/m^2
aspect_ratio = AR = 7
gross_weight = 24 # N
payload_weight_fraction = f = 0.42
length_width_ratio = LB = 0.8
B = 1
S = gross_weight / wing_loading # m^2
flight_score = 20 * f * gross_weight + f - ((AR * S)**2 * (1 + LB)**4) + 20 * B
print(flight_score)

exit()

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_theme()
# c1 = 20.
# c2 = 1.
# c3 = -1.
# c4 = 20.
# AR = 8
# W_S = 80
# L_b = 0.8
# Wr = 1.
# Dr = 1.
# Wp_W0 = 0.3
# # -----
# num = 100
# W0 = np.linspace(0., 35., num)
# # -----
# Wp = Wp_W0 * W0
# S = W0 / W_S
# b = np.sqrt(AR * S)
# L = L_b * b
# D = L + b
# score = c1 * Wp / Wr + c3 * (D / Dr) ** 4.
# dD_dW0 = (1 + L_b) * np.sqrt(AR / W_S) * 0.5 / W0 ** 0.5
# dscore_dW0 = c1 * Wp_W0 / Wr + c3 * (D / Dr) ** 4. * 4 / D * dD_dW0
# dscore_dW0 = c1 * Wp_W0 / Wr + c3 * 2 * W0 * AR ** 2 / W_S ** 2 / Dr ** 4 * (1 + L_b) ** 4
# plt.figure(figsize=(6., 3.))
# plt.plot(W0 / 9.81, score, label='scoring function')
# plt.plot(W0 / 9.81, dscore_dW0, label='scoring function deriv.')
# plt.xlabel('W0 [kg]')
# plt.savefig('scoring.pdf')
# plt.show()
# print(score)