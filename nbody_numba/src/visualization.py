import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider # <-- New Import for the Slider!

# --- Matplotlib Backend Handling ---
# We need a true interactive backend for widgets
try:
    # TkAgg is a good cross-platform choice for widgets
    matplotlib.use('TkAgg') 
except Exception:
    print("Warning: TkAgg backend failed. Interactive widgets may not work.")
    # Fallback to Agg, but interactivity will fail.
    matplotlib.use('Agg')

# ... (The rest of your imports and other visualization functions go here) ...


def animate_simulation(positions_history):
    """
    Creates an interactive 3D animation with a speed control slider.
    
    Args:
        positions_history (np.ndarray): Shape (n_steps, n_particles, 3)
    """
    
    n_steps, _, _ = positions_history.shape
    
    # 1. Setup the Figure and Axes
    fig = plt.figure(figsize=(10, 8))
    # Create the main 3D axes at the top of the figure
    ax = fig.add_subplot(211, projection='3d') # <-- Changed to 2 rows, 1 col, 1st plot
    
    # Create the secondary axes for the slider at the bottom
    # [left, bottom, width, height] relative to the figure
    ax_slider = fig.add_axes([0.2, 0.02, 0.6, 0.03]) 
    
    # Set initial plotting parameters
    initial_data = positions_history[0]
    scatter = ax.scatter(initial_data[:, 0], initial_data[:, 1], initial_data[:, 2], s=5)
    
    # --- Axis Limits ---
    min_val = np.min(positions_history)
    max_val = np.max(positions_history)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])
    ax.set_xlabel("X Position")
    
    # 2. Create the Slider Widget
    # The slider controls the frame delay (interval) in milliseconds.
    # We invert it so a higher slider value means a slower animation (longer delay).
    speed_slider = Slider(
        ax=ax_slider,
        label='Animation Speed (ms delay)',
        valmin=1,           # Minimum delay (fastest)
        valmax=200,         # Maximum delay (slowest)
        valinit=30,         # Default delay (30 ms)
        valstep=1
    )
    
    # 3. Define Update Function
    def update(frame):
        data = positions_history[frame]
        scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
        ax.set_title(f"N-Body Simulation: Timestep {frame}")
        return scatter,

    # 4. Create Animation Object (uses default interval)
    anim = FuncAnimation(fig, update, frames=n_steps, 
                         interval=speed_slider.valinit, blit=False, repeat=True)

    # 5. Define Slider Callback
    def update_speed(val):
        """Updates the animation's interval (speed) when the slider is moved."""
        # FuncAnimation uses the interval parameter to set the delay between frames
        anim.event_source.interval = val
        # This forces the animation to redraw immediately with the new interval
        fig.canvas.draw_idle()

    # Link the slider's value change event to our update function
    speed_slider.on_changed(update_speed)

    # Show the interactive figure
    plt.show()

def plot_snapshot(positions, title="N-Body Simulation Snapshot"):
    """
    Creates a static 3D scatter plot of particle locations.

    Args:
        positions (np.ndarray): An (N, 3) array of particle [x, y, z] locations.
        title (str): The title for the plot.
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]

    ax.scatter(x_coords, y_coords, z_coords, s=40)

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title(title)

    min_val = np.min(positions)
    max_val = np.max(positions)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.set_zlim([min_val, max_val])
    ax.set_aspect('auto') # 'auto' is common, 'equal' can be buggy in 3D

    plt.show()
