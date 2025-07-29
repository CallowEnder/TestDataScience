import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constrained Optimization:
# The core problem involves optimizing (finding the maximum or minimum of) a function, say f(x, y, z)
# under a constraint defined by another function g(x, y) = 0.
# Here, we will visualize the function f and the constraint g in a 3D space using surface plots. 
# The function f will be plotted as a surface, and the constraint g will be visualized as a curve
# on the surface of f. 

# To address the constraints, a new function called the Lagrangian is formed. 
# For a single constraint g(x, y, z) = 0, the Lagrangian is defined as:
# L(x, y, z, λ) = f(x, y, z) + λg(x, y, z), where λ is the Lagrange multiplier:
# (i.e., ∂L/∂g = λ).  

# In this example the constraint g(x, y) = 0 is a hyperbolic paraboloid.
# The function f(x, y) = -exp(x - y^2 + xy) is a negative exponential function.

# Import necessary libraries
# Define the functions
import matplotlib.pyplot as plt
def f(x, y):
    """Function to minimize."""
    return -np.exp(x - y**2 + x*y)

def g(x, y):
    """Constraint function g(x, y) = 0."""
    return np.cosh(y) + x - 2

# Create a grid of points
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Compute function values
Z1 = f(X, Y)
Z2 = g(X, Y)

fig = plt.figure(figsize=(18, 6))

# Surface plot for f
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax1.set_title('Surface plot of f(x, y)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')
ax1.set_zlim(np.min(Z1), np.max(Z1))  # Fix z-axis limits for f

# Surface plot for g
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
ax2.set_title('Surface plot of g(x, y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('g(x, y)')
ax2.set_zlim(np.min(Z2), np.max(Z2))  # Fix z-axis limits for g

# Fix z-axis limits for g surface
# Combined plot: f surface + constraint curve + projection onto g surface
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.7)

tol = 0.05
mask = np.abs(Z2) < tol

# Constraint curve on f surface (red)
ax3.scatter(X[mask], Y[mask], Z1[mask], color='red', s=15, label='g(x,y)=0 on f surface')

# Project constraint curve onto g surface (blue)
ax3.scatter(X[mask], Y[mask], Z2[mask], color='blue', s=15, label='g(x,y)=0 on g surface')

# Draw vertical lines connecting points on f surface and g surface to emphasize projection
for xi, yi, zf, zg in zip(X[mask], Y[mask], Z1[mask], Z2[mask]):
    # Draw line from f surface point down/up to g surface point (on g surface plot)
    ax3.plot([xi, xi], [yi, yi], [zf, zg], color='gray', alpha=0.5, linewidth=0.7)

ax3.set_title('f(x,y) with constraint g(x,y)=0 and projection')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('Function values')
# Set z limits to cover both surfaces reasonably
z_min = min(np.min(Z1), np.min(Z2))
z_max = max(np.max(Z1), np.max(Z2))
ax3.set_zlim(z_min, z_max)
ax3.legend()

# ... existing code ...

plt.tight_layout()
plt.show()