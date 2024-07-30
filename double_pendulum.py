import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

# Constants
g = 9.81  # Gravity (m/s^2)
L1, L2 = 1.0, 1.0  # Lengths of the pendulum arms (meters)
m1, m2 = 1.0, 1.0  # Masses at the ends of the pendulum arms (kilograms)
num_pendulums = 1000  # Number of pendulums


# Differential equations for double pendulum
def double_pendulum(t, y):
    dydt = []
    for i in range(num_pendulums):
        offset = 4 * i
        theta1, omega1, theta2, omega2 = y[offset:offset + 4]
        c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)


        denom1 = (m1 + m2) * L1 - m2 * L1 * c ** 2
        theta1_acc = (m2 * g * np.sin(theta2) * c - m2 * s * (L2 * omega2 ** 2 + L1 * omega1 ** 2 * c) - (
                    m1 + m2) * g * np.sin(theta1)) / denom1

        denom2 = (L2 / L1) * denom1
        theta2_acc = ((m1 + m2) * (L1 * omega1 ** 2 * s - g * np.sin(theta2) + g * np.sin(
            theta1) * c) + m2 * L2 * omega2 ** 2 * s * c) / denom2

        dydt.extend([omega1, theta1_acc, omega2, theta2_acc])
    return dydt


# Initial conditions
initial_conditions = []
base_angle = np.pi  # Base angle for initial condition
delta = 0.001  #  variation between initial conditions
for i in range(num_pendulums):
    initial_conditions.extend([
        base_angle + i * delta, 0,
        base_angle - i * delta, 0
    ])



# Time span for the simulation
t_span = [0, 20 // 5]
t_eval = np.linspace(*t_span, 1000 // 5)

# Solve the ODEs
solution = solve_ivp(double_pendulum, t_span, initial_conditions, t_eval=t_eval, method='RK45')

# Setup the plot 
fig, ax = plt.subplots()
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2)
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg-6.1-essentials_build/ffmpeg-6.1-essentials_build/bin/ffmpeg.exe'

# Create line collections for each pendulum with fading effect
collections = []
colors = plt.cm.cool(np.linspace(0, 1, num_pendulums))  # Color map
for i in range(num_pendulums):
    offset = i * 4
    x2 = L1 * np.sin(solution.y[offset]) + L2 * np.sin(solution.y[offset + 2])
    y2 = -L1 * np.cos(solution.y[offset]) - L2 * np.cos(solution.y[offset + 2])
    points = np.array([x2, y2]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=0.5, color=colors[i], alpha=0.6)
    ax.add_collection(lc)
    collections.append((lc, segments))


# Update function for the animation
def update(frame):
    num_visible = 30 # Reduced number of visible frames for faster fading
    for lc, segments in collections:
        start = max(0, frame - num_visible)
        lc.set_segments(segments[start:frame + 1])
        # Faster alpha fading
        alphas = np.linspace(0, 1, frame - start + 1)
        lc.set_alpha(np.mean(alphas))  # Apply an average alpha for fading
    return [lc[0] for lc in collections]


# Create the animation
ani = FuncAnimation(fig, update, frames=len(t_eval), blit=True, interval=100, repeat=False)
ani.save(filename="animations/pendulums1000/double_pendulum1000_long.gif", writer='pillow', dpi=300, fps=125) # dpi = 300

# Show the plot
# plt.title('Multiple Double Pendulums with Faster Fading Trajectories')
# plt.xlabel('X position (m)')
# plt.ylabel('Y position (m)')
# plt.axis('equal')
# plt.grid(True)
# plt.show()
