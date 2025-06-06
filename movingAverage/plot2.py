import numpy as np
import matplotlib.pyplot as plt

# Adjust these filenames if needed
x = np.fromfile("h_xmean.bin", dtype=np.float32)
y = np.fromfile("h_xmean_gpu.bin", dtype=np.float32)

plt.figure(figsize=(12, 6))
plt.plot(x, label="cpu")
plt.plot(y, label="gpu)")
plt.legend()
plt.title("Float Dump Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

gold = np.fromfile("h_y_gold.bin", dtype=np.float32)
x = np.fromfile("h_y.bin", dtype=np.float32)
y = np.fromfile("h_y_gpu.bin", dtype=np.float32)

plt.figure(figsize=(12, 6))
plt.plot(x, label="cpu")
plt.plot(y, label="gpu")
plt.plot(gold, label="gold")
plt.legend()
plt.title("Float Dump Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()