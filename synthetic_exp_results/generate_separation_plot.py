import matplotlib.pyplot as plt 
import numpy as np

x = np.arange(.21,.3,.01)
y = [0.5, 1.0, 0.8999999999999999, 0.7, 0.8, 0.5499999999999999,0.95,0.75, 1.0]

plt.plot(x, y)
plt.xlabel("spurious_p")
plt.ylabel("high vs low sample scores separation")
plt.title("% of spurious samples vs. score separability")
plt.tight_layout()
plt.savefig("exp_2.png")