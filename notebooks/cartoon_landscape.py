import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up grid
x = np.linspace(-2, 2, 300)
y = np.linspace(-0.4, 0.4, 300)
X, Y = np.meshgrid(x, y)

# Define a rough energy landscape function
# z ~ "description length" of models
Z = (
    np.sin(3*X) * np.cos(5*Y)
    + 0.3 * np.sin(7*X*Y)
    + 0.2 * np.cos(10*Y)
    + 0.001 * np.random.normal(size=X.shape)  # add roughness / noise
)

# Add a large-scale trend so there's a global minimum / maximum
Z += 0.25 * (X**2 + 2*Y**2)

# Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='bwr', linewidth=0, antialiased=True, alpha=0.9)

# Labels
#ax.set_xlabel(r'$x$',size=24)
#ax.set_ylabel(r'$y$',size=24)
ax.set_zlabel(r'$\mathcal{L}$, Description length',size=22)
#ax.set_title("Symbolic Regression Energy Landscape", fontsize=14)

# Aesthetics
ax.view_init(elev=35, azim=120)
#fig.colorbar(surf, shrink=0.5, aspect=10, label='Energy')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


plt.tight_layout()
output_path='../results/paper_figures/figure1/'
plt.savefig(output_path +'landscape' +'.svg',dpi=300)


plt.show()

