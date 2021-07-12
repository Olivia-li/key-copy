import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.plot(x, y)
plt.grid([True, 'major', 'both'], linewidth=1.5, color="black")
plt.xlim([0, 10])
plt.ylim([0, 10])
ax.set_xticks(range(1, 10), minor=False)
ax.set_yticks(range(1, 10), minor=False)
ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')
ax.set_aspect("equal")

# Add black marker on top-left
plt.plot(0.5, 9.5, 'ks', markersize=55)

plt.savefig("key_grid.png")
