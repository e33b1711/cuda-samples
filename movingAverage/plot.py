import numpy as np
import matplotlib.pyplot as plt

# Adjust these filenames if needed
x = np.fromfile("h_x.bin", dtype=np.float32)
y = np.fromfile("h_y.bin", dtype=np.float32)

plt.figure(figsize=(12, 6))
plt.plot(x, label="Input (h_x)")
plt.plot(y, label="Moving Average (h_y)")
plt.legend()
plt.title("Float Dump Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()