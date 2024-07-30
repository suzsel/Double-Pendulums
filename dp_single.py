import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# Constants
g = 9.81  # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths of the pendulum arms (m)
m1, m2 = 1.0, 1.0  # Masses at the ends of the pendulum arms (kg)


# Equations of motion derived from Lagrange's equations
def double_pendulum(t, y):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1_dot = z1
    theta2_dot = z2

    denom1 = (m1 + m2) * L1 - m2 * L1 * c ** 2
    theta1_acc = (m2 * g * np.sin(theta2) * c - m2 * s * (L2 * z2 ** 2 + L1 * z1 ** 2 * c) - (m1 + m2) * g * np.sin(
        theta1)) / denom1

    denom2 = (L2 / L1) * denom1
    theta2_acc = ((m1 + m2) * (
                L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) + m2 * L2 * z2 ** 2 * s * c) / denom2

    return [theta1_dot, theta1_acc, theta2_dot, theta2_acc]


# Initial conditions
initial_conditions = [np.pi / 2, 0, np.pi / 2, 0]

# Time span for the simulation
t_span = [0, 20]
t_eval = np.linspace(*t_span, 500)

# Solve the ODE
solution = solve_ivp(double_pendulum, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Unpack the solution
theta1, theta2 = solution.y[0], solution.y[2]

# Calculate the Cartesian coordinates of the pendulum bobs
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Setup the figure and axis for animation
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2)

# Prepare to draw line segments with varying properties
points = np.array([x2, y2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='autumn')
lc.set_array(np.linspace(0, 1, len(x2)))
ax.add_collection(lc)


# Update function for the animation
def update(frame):
    num_visible = 100  # Number of visible points at any time
    start = max(0, frame - num_visible)
    end = frame + 1

    if frame < num_visible:
        # Early frames, gradually increase the length of the visible line
        segment_subset = segments[:end]
        alpha_values = np.linspace(0.1, 1, end)
    else:
        # Later frames, keep the line length constant but shift
        segment_subset = segments[start:end]
        alpha_values = np.linspace(0.1, 1, num_visible)

    lc.set_segments(segment_subset)
    lc.set_linewidths(np.linspace(0.5, 5, len(segment_subset)))
    lc.set_alpha(np.mean(alpha_values))  # Apply an average alpha for simplicity

    return lc,


# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), blit=True, interval=20)

# Show the animation
plt.title('Double Pendulum Trajectory with Fading and Thickening Effect')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.grid(True)
plt.axis('equal')
plt.show()
