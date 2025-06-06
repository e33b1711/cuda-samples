import numpy as np
import matplotlib.pyplot as plt

# Adjust these filenames if needed
x = np.fromfile("h_x.bin", dtype=np.float32)
mean = np.fromfile("h_xmean.bin", dtype=np.float32)
y = np.fromfile("h_y.bin", dtype=np.float32)

plt.figure(figsize=(12, 6))
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.plot(mean, label="mean")
plt.legend()
plt.title("Float Dump Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

x = np.fromfile("h_x.bin", dtype=np.float32)
gold = np.fromfile("h_y_gold.bin", dtype=np.float32)
cpu = np.fromfile("h_y.bin", dtype=np.float32)
gpu = np.fromfile("h_y_gpu.bin", dtype=np.float32)

plt.figure(figsize=(12, 6))
plt.plot(x, label="x")
plt.plot(cpu, label="cpu")
plt.plot(gpu, label="gpu")
plt.plot(gold, label="gold")
plt.legend()
plt.title("Float Dump Plot")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()