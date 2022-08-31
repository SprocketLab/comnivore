import matplotlib.pyplot as plt 
import numpy as np

x = np.arange(.1,.61,.1)
y_no_weight = [0.962, 0.855, 0.708, 0.547, 0.407, 0.263]
y_weight = [0.998, 0.984, 0.981, 0.921, 0.963, 0.781]

plt.plot(x, y_no_weight, label="no weight")
plt.plot(x, y_weight, label="with weight")
plt.xlabel("spurious_p")
plt.ylabel("test acc (%)")
plt.title("% of spurious samples vs. test acc")
plt.tight_layout()
plt.legend()
plt.savefig("test_acc_1.png")