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

def generate_solar_system(n_bodies=100):
    """
    Generates a stable 'Solar System' like setup for showcasing.
    Body 0 is a massive 'Sun'. The rest are planets orbiting it.
    """
    pos = np.zeros((n_bodies, 3), dtype=np.float64)
    vel = np.zeros((n_bodies, 3), dtype=np.float64)
    mass = np.random.rand(n_bodies).astype(np.float64) * 1e10  # Asteroids
    
    # 1. Setup the Sun (Massive, at center)
    mass[0] = 1.0e20 
    pos[0] = [0, 0, 0]
    vel[0] = [0, 0, 0]
    
    # 2. Setup Planets/Asteroids
    for i in range(1, n_bodies):
        # Random distance from 50 to 200
        dist = 50 + np.random.rand() * 150
        
        # Random angle
        theta = np.random.rand() * 2 * np.pi
        
        # Position (Flat disk for nicer visualization)
        pos[i, 0] = dist * np.cos(theta)
        pos[i, 1] = dist * np.sin(theta)
        pos[i, 2] = (np.random.rand() - 0.5) * 5 # Small Z variation
        
        # Velocity for circular orbit: v = sqrt(GM / r)
        # Vector direction: tangent to the circle (-sin, cos)
        v_orb = np.sqrt(6.6743e-11 * mass[0] / dist)
        vel[i, 0] = -v_orb * np.sin(theta)
        vel[i, 1] = v_orb * np.cos(theta)
        vel[i, 2] = 0
        
        # Make a few heavy planets
        if i < 5:
            mass[i] *= 1000  # Gas giants
            
    return pos, vel, mass

def visualize_showcase(history, masses, tail_length=40):
    """
    High-quality animation with trails and auto-centering.
    """
    n_steps, n_bodies, _ = history.shape
    
    # 1. Setup Style
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove panes for "Space" look
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.grid(False)
    
    # 2. visual settings
    # Normalize masses for size: Sun is huge, others are small
    sizes = np.log(masses) 
    sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    sizes = 5 + sizes * 100  # Scale between 5 and 105
    
    # Prepare color map (based on particle index so we can track them)
    colors = plt.cm.plasma(np.linspace(0, 1, n_bodies))
    
    # Initial Plot objects
    # We use plot() for trails and scatter() for heads
    trails = [ax.plot([], [], [], '-', lw=0.5, alpha=0.5, color=colors[i])[0] for i in range(n_bodies)]
    heads = ax.scatter(history[0,:,0], history[0,:,1], history[0,:,2], s=sizes, c=colors)
    
    # Camera centering strategy
    # Center on the heaviest body (The Sun)
    center_idx = np.argmax(masses)
    
    # Calculate global bounds to fix the camera zoom
    max_range = np.max(np.abs(history[-1] - history[-1][center_idx])) * 1.1
    
    def update(frame):
        # 1. Update Heads (Scatter)
        heads._offsets3d = (history[frame,:,0], history[frame,:,1], history[frame,:,2])
        
        # 2. Update Trails (Lines)
        # Only draw history up to 'tail_length' frames back
        start = max(0, frame - tail_length)
        
        for i in range(n_bodies):
            # Optimization: Only draw trails for significant bodies or if N is small
            # If N > 200, maybe skip trails for tiny asteroids
            if masses[i] > masses.min() * 10 or n_bodies < 100:
                trails[i].set_data(history[start:frame+1, i, 0], history[start:frame+1, i, 1])
                trails[i].set_3d_properties(history[start:frame+1, i, 2])
        
        # 3. Dynamic Camera (Follow the Sun)
        center = history[frame, center_idx]
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_title(f"Time Step: {frame}")
        return trails + [heads]

    anim = FuncAnimation(fig, update, frames=n_steps, interval=20, blit=False)
    plt.show()

# --- RUN THE SHOWCASE ---
if __name__ == "__main__":
    from simulation import run_simulation

    # 1. Generate a stable Solar System
    N = 100
    pos, vel, mass = generate_solar_system(N)
    
    # 2. Run Simulation (Import your run_simulation function first!)
    # Note: Use a smaller dt for stability in this tight system
    # Import the function from your previous code block
    # from your_script_name import run_simulation 
    
    print("Simulating physics...")
    # Assuming run_simulation is available from previous context:
    history = run_simulation(pos, vel, mass, dt=0.5, steps=400, device="auto")
    
    # 3. Visualize
    print("Rendering animation...")
    visualize_showcase(history, mass, tail_length=50)