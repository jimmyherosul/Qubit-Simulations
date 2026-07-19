# ----------------------------- Import libraries -----------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy import integrate

# ----------------------------- Parameters -----------------------------
hbar = 1
gamma = 2/hbar
V_abs = 2
lim = 10
t = np.linspace(0, 3.2, 500)
interval = 10       # Time between frames in milliseconds

# Magnetic field vector in the laboratory frame
Bz_lab = 12
Omega_0 = gamma * Bz_lab                   # Larmor angular frequency (rate at which M_lab precesses relative to B_lab due to the static B_lab-field)
Omega_d = Omega_0                          # Drive angular frequency set to the Larmor angular frequency for maximal driven precession effect (at resonance)
#Bx_lab = np.zeros_like(t)          # For free precession under only static Bz_lab field
#By_lab = np.zeros_like(t)          # For free precession under only static Bz_lab field
#transverse_drive_on = False        # For free precession under only static Bz_lab field
Bx_lab = -V_abs*np.cos(Omega_d*t)          # For driven precession under transverse rotating/oscillating fields
By_lab = V_abs*np.sin(Omega_d*t)           # For driven precession under transverse rotating/oscillating fields
transverse_drive_on = True                 # For driven precession under transverse rotating/oscillating fields  

B_lab_initial = [Bx_lab[0], By_lab[0], Bz_lab]
B_lab_mag = np.linalg.norm(B_lab_initial)

# Magnetization vector
Mz0_lab = 6     #0 for driven precession but with no sprialing motion in xy-plane
Mx0_lab = 2          
My0_lab = 2     
M_initial_lab = [Mx0_lab, My0_lab, Mz0_lab]

# =============================================================================
#                           LABORATORY FRAME SOLUTION
# =============================================================================

# Construct the coupled Bloch equations as a system of first-order ODEs from the cross product of M_lab and B_lab(t) in laboratory frame
# dM/dt = gamma * M_lab x B_lab(t)
def f_lab(time, y):
    M_lab_current = y

    Bx_current = np.interp(time, t, Bx_lab)
    By_current = np.interp(time, t, By_lab)
    B_lab_current = np.array([Bx_current, By_current, Bz_lab])

    dMdt = gamma * np.cross(M_lab_current, B_lab_current)

    return dMdt

# Solve the ODEs using initial value problem
# Note: Python's ivp solver uses the 4th/5th order Runge-Kutta method (RK45) by default, which is more accurate than the standard Euler's method
solution_lab = integrate.solve_ivp(
    fun=f_lab,
    t_span=(t[0], t[-1]),
    y0=M_initial_lab,
    t_eval=t
)

Mx_lab = solution_lab.y[0]
My_lab = solution_lab.y[1]
Mz_lab = solution_lab.y[2]

M_lab_initial = [Mx_lab[0], My_lab[0], Mz_lab[0]]
M_lab_mag = np.linalg.norm(M_lab_initial)

if M_lab_mag == 0:
    M_lab_mag = 1e-12

# =============================================================================
#                           ROTATING FRAME SOLUTION
# =============================================================================

# Rotating-frame equation:
# dM/dt = M x [ (-2|V|/hbar) i_x' + (Omega_0 - Omega_d) k_z' ]
#
# The vector inside the brackets has units of angular frequency.
# Since dM/dt = gamma * M x B_eff, we can write:
#
# Omega_eff_rot = gamma * B_eff_rot
# B_eff_rot = Omega_eff_rot / gamma

if transverse_drive_on:
    Omega_eff_rot = np.array([
        -2 * V_abs / hbar,
        0,
        Omega_0 - Omega_d
    ])
else:
    Omega_eff_rot = np.array([
        0,
        0,
        Omega_0 - Omega_d
    ])

B_eff_rot = Omega_eff_rot / gamma

B_eff_x_rot = np.full_like(t, B_eff_rot[0])
B_eff_y_rot = np.full_like(t, B_eff_rot[1])

B_eff_rot_mag = np.linalg.norm(B_eff_rot)

if B_eff_rot_mag == 0:
    B_eff_rot_mag = 1e-12

def f_rot(time, y):
    M_rot_current = y

    # Equivalent forms:
    # dMdt = np.cross(M_rot_current, Omega_eff_rot)
    # dMdt = gamma * np.cross(M_rot_current, B_eff_rot)

    dMdt = gamma * np.cross(M_rot_current, B_eff_rot)

    return dMdt

solution_rot = integrate.solve_ivp(
    fun=f_rot,
    t_span=(t[0], t[-1]),
    y0=M_initial_lab,
    t_eval=t
)

Mx_rot = solution_rot.y[0]
My_rot = solution_rot.y[1]
Mz_rot = solution_rot.y[2]

M_rot_initial = [Mx_rot[0], My_rot[0], Mz_rot[0]]
M_rot_mag = np.linalg.norm(M_rot_initial)

if M_rot_mag == 0:
    M_rot_mag = 1e-12

# =============================================================================
#                           FIGURE 1: LABORATORY FRAME
# =============================================================================

fig_lab = plt.figure(figsize=(12, 5))
plt.figure(fig_lab.number)
plt.subplots_adjust(bottom=0.25)

# ----------------------------- Lab-frame time plot -----------------------------
ax_time_lab = fig_lab.add_subplot(1, 2, 1)
ax_time_lab.set_xlim([min(t), max(t)])
ax_time_lab.set_ylim([-1.2 * V_abs, 1.2 * V_abs])

time_plot_Bx_lab, = ax_time_lab.plot(
    [],
    [],
    label=r'$B_x^{lab}(t) = -|V|\cos(\Omega_d t)$'
)

time_plot_By_lab, = ax_time_lab.plot(
    [],
    [],
    label=r'$B_y^{lab}(t) = |V|\sin(\Omega_d t)$'
)

time_cursor_point_Bx_lab, = ax_time_lab.plot(
    [],
    [],
    "ko",
    markersize=6,
    label=r"Current point"
)

time_cursor_point_By_lab, = ax_time_lab.plot(
    [],
    [],
    "ko",
    markersize=6
)

ax_time_lab.set_xlabel('Time')
ax_time_lab.set_ylabel('Field amplitude')
ax_time_lab.set_title('Transverse B-Field Components in Laboratory Frame')

ax_time_lab.minorticks_on()
ax_time_lab.grid(True, which='major', linestyle='-', linewidth=0.6)
ax_time_lab.grid(True, which='minor', linestyle='--', linewidth=0.3)
ax_time_lab.legend(loc="upper right")

# ----------------------------- Lab-frame 3D vector plot -----------------------------
ax_3d_lab = fig_lab.add_subplot(1, 2, 2, projection="3d")

M_lab_arrow_head_size = (0.15 / 5 * lim) / M_lab_mag * np.sqrt(12)
B_lab_arrow_head_size = (0.15 / 5 * lim) / B_lab_mag * np.sqrt(12)

B_lab_arrow = ax_3d_lab.quiver(
    0, 0, 0,
    B_lab_initial[0], B_lab_initial[1], B_lab_initial[2],
    color="blue",
    arrow_length_ratio=B_lab_arrow_head_size,
    label=r"$B^{lab}(t)$"
)

M_lab_arrow = ax_3d_lab.quiver(
    0, 0, 0,
    M_lab_initial[0], M_lab_initial[1], M_lab_initial[2],
    color="magenta",
    arrow_length_ratio=M_lab_arrow_head_size,
    label=r"$M^{lab}(t)$"
)

ax_3d_lab.set_xlim([-lim, lim])
ax_3d_lab.set_ylim([-lim, lim])
ax_3d_lab.set_zlim([-lim, lim])

ax_3d_lab.plot([-lim, lim], [0, 0], [0, 0], color="black")
ax_3d_lab.plot([0, 0], [-lim, lim], [0, 0], color="black")
ax_3d_lab.plot([0, 0], [0, 0], [-lim, lim], color="black")

ax_3d_lab.set_xlabel("$X$")
ax_3d_lab.set_ylabel("$Y$")
ax_3d_lab.set_zlabel("$Z$")
ax_3d_lab.set_title("3D Vector Plot in Laboratory Frame")

B_lab_cursor_point, = ax_3d_lab.plot(
    [B_lab_initial[0]],
    [B_lab_initial[1]],
    [B_lab_initial[2]],
    "ko",
    markersize=6,
    label=r"Current point of $B^{lab}(t)$"
)

M_lab_cursor_point, = ax_3d_lab.plot(
    [M_lab_initial[0]],
    [M_lab_initial[1]],
    [M_lab_initial[2]],
    "ro",
    markersize=6,
    label=r"Current point of $M^{lab}(t)$"
)

M_lab_trace, = ax_3d_lab.plot(
    [],
    [],
    [],
    "r-",
    linewidth=1.5
)

ax_3d_lab.legend(loc="upper right")

# ----------------------------- Animating the plots in lab frame -----------------------------
def update_data_lab(frame):
    global B_lab_arrow, M_lab_arrow

    time_plot_Bx_lab.set_data(
        t[:frame + 1],
        Bx_lab[:frame + 1]
    )

    time_plot_By_lab.set_data(
        t[:frame + 1],
        By_lab[:frame + 1]
    )

    time_cursor_point_Bx_lab.set_data(
        [t[frame]],
        [Bx_lab[frame]]
    )

    time_cursor_point_By_lab.set_data(
        [t[frame]],
        [By_lab[frame]]
    )

    B_lab_arrow.remove()
    M_lab_arrow.remove()

    B_lab_current = [
        Bx_lab[frame],
        By_lab[frame],
        Bz_lab
    ]

    B_lab_mag_current = np.linalg.norm(B_lab_current)

    if B_lab_mag_current == 0:
        B_lab_mag_current = 1e-12

    B_lab_arrow_head_size = (0.15 / 5 * lim) / B_lab_mag_current * np.sqrt(12)

    B_lab_arrow = ax_3d_lab.quiver(
        0, 0, 0,
        B_lab_current[0], B_lab_current[1], B_lab_current[2],
        color="blue",
        arrow_length_ratio=B_lab_arrow_head_size
    )

    M_lab_current = [
        Mx_lab[frame],
        My_lab[frame],
        Mz_lab[frame]
    ]

    M_lab_mag_current = np.linalg.norm(M_lab_current)

    if M_lab_mag_current == 0:
        M_lab_mag_current = 1e-12

    M_lab_arrow_head_size = (0.15 / 5 * lim) / M_lab_mag_current * np.sqrt(12)

    M_lab_arrow = ax_3d_lab.quiver(
        0, 0, 0,
        M_lab_current[0], M_lab_current[1], M_lab_current[2],
        color="magenta",
        arrow_length_ratio=M_lab_arrow_head_size
    )

    B_lab_cursor_point.set_data(
        [B_lab_current[0]],
        [B_lab_current[1]]
    )
    B_lab_cursor_point.set_3d_properties([B_lab_current[2]])

    M_lab_cursor_point.set_data(
        [M_lab_current[0]],
        [M_lab_current[1]]
    )
    M_lab_cursor_point.set_3d_properties([M_lab_current[2]])

    M_lab_trace.set_data(
        Mx_lab[:frame + 1],
        My_lab[:frame + 1]
    )
    M_lab_trace.set_3d_properties(Mz_lab[:frame + 1])

    return (
        time_plot_Bx_lab,
        time_plot_By_lab,
        time_cursor_point_Bx_lab,
        time_cursor_point_By_lab,
        B_lab_arrow,
        M_lab_arrow,
        B_lab_cursor_point,
        M_lab_cursor_point,
        M_lab_trace
    )

animation_lab = FuncAnimation(
    fig=fig_lab,
    func=update_data_lab,
    frames=len(t),
    interval=interval,
    blit=False
)

# =============================================================================
#                           FIGURE 2: ROTATING FRAME
# =============================================================================

fig_rot = plt.figure(figsize=(12, 5))
plt.figure(fig_rot.number)
plt.subplots_adjust(bottom=0.25)

# ----------------------------- Rotating-frame effective B-field time plot -----------------------------
ax_time_rot = fig_rot.add_subplot(1, 2, 1)

rot_y_min = min(
    np.min(B_eff_x_rot),
    np.min(B_eff_y_rot)
)

rot_y_max = max(
    np.max(B_eff_x_rot),
    np.max(B_eff_y_rot)
)

if rot_y_min == rot_y_max:
    rot_y_min -= 1
    rot_y_max += 1
else:
    margin = 0.2 * (rot_y_max - rot_y_min)
    rot_y_min -= margin
    rot_y_max += margin

ax_time_rot.set_xlim([min(t), max(t)])
ax_time_rot.set_ylim([rot_y_min, rot_y_max])

time_plot_B_eff_x_rot, = ax_time_rot.plot(
    [],
    [],
    label=r'$B_{eff,x^\prime}^{rot} = -|V|$'
)

time_plot_B_eff_y_rot, = ax_time_rot.plot(
    [],
    [],
    label=r'$B_{eff,y^\prime}^{rot} = 0$'
)

time_cursor_point_B_eff_x_rot, = ax_time_rot.plot(
    [],
    [],
    "ko",
    markersize=6,
    label="Current point"
)

time_cursor_point_B_eff_y_rot, = ax_time_rot.plot(
    [],
    [],
    "ko",
    markersize=6
)

ax_time_rot.set_xlabel('Time')
ax_time_rot.set_ylabel('Field amplitude')
ax_time_rot.set_title('Effective Transverse B-Field Components in Rotating Frame')

ax_time_rot.minorticks_on()
ax_time_rot.grid(True, which='major', linestyle='-', linewidth=0.6)
ax_time_rot.grid(True, which='minor', linestyle='--', linewidth=0.3)
ax_time_rot.legend(loc="upper right")

# ----------------------------- Rotating-frame 3D vector plot -----------------------------
ax_3d_rot = fig_rot.add_subplot(1, 2, 2, projection="3d")

M_rot_arrow_head_size = (0.15 / 5 * lim) / M_rot_mag * np.sqrt(12)
B_eff_rot_arrow_head_size = (0.15 / 5 * lim) / B_eff_rot_mag * np.sqrt(12)

B_eff_rot_arrow = ax_3d_rot.quiver(
    0, 0, 0,
    B_eff_rot[0], B_eff_rot[1], B_eff_rot[2],
    color="blue",
    arrow_length_ratio=B_eff_rot_arrow_head_size,
    label=r"$B_{\mathrm{eff}}^{rot}$"
)

M_rot_arrow = ax_3d_rot.quiver(
    0, 0, 0,
    M_rot_initial[0], M_rot_initial[1], M_rot_initial[2],
    color="magenta",
    arrow_length_ratio=M_rot_arrow_head_size,
    label=r"$M^{rot}(t)$"
)

ax_3d_rot.set_xlim([-lim, lim])
ax_3d_rot.set_ylim([-lim, lim])
ax_3d_rot.set_zlim([-lim, lim])

ax_3d_rot.plot([-lim, lim], [0, 0], [0, 0], color="black")
ax_3d_rot.plot([0, 0], [-lim, lim], [0, 0], color="black")
ax_3d_rot.plot([0, 0], [0, 0], [-lim, lim], color="black")

ax_3d_rot.set_xlabel("$X'$")
ax_3d_rot.set_ylabel("$Y'$")
ax_3d_rot.set_zlabel("$Z'$")
ax_3d_rot.set_title("3D Vector Plot in Rotating Frame")

B_eff_rot_cursor_point, = ax_3d_rot.plot(
    [B_eff_rot[0]],
    [B_eff_rot[1]],
    [B_eff_rot[2]],
    "ko",
    markersize=6,
    label=r"Current point of $B_{\mathrm{eff}}^{rot}$"
)

M_rot_cursor_point, = ax_3d_rot.plot(
    [M_rot_initial[0]],
    [M_rot_initial[1]],
    [M_rot_initial[2]],
    "ro",
    markersize=6,
    label=r"Current point of $M^{rot}(t)$"
)

M_rot_trace, = ax_3d_rot.plot(
    [],
    [],
    [],
    "r-",
    linewidth=1.5
)

ax_3d_rot.legend(loc="upper right")

# ----------------------------- Animating the plots in rotating frame -----------------------------
def update_data_rot(frame):
    global B_eff_rot_arrow, M_rot_arrow

    time_plot_B_eff_x_rot.set_data(
        t[:frame + 1],
        B_eff_x_rot[:frame + 1]
    )

    time_plot_B_eff_y_rot.set_data(
        t[:frame + 1],
        B_eff_y_rot[:frame + 1]
    )

    time_cursor_point_B_eff_x_rot.set_data(
        [t[frame]],
        [B_eff_x_rot[frame]]
    )

    time_cursor_point_B_eff_y_rot.set_data(
        [t[frame]],
        [B_eff_y_rot[frame]]
    )

    B_eff_rot_arrow.remove()
    M_rot_arrow.remove()

    B_eff_rot_arrow = ax_3d_rot.quiver(
        0, 0, 0,
        B_eff_rot[0], B_eff_rot[1], B_eff_rot[2],
        color="blue",
        arrow_length_ratio=B_eff_rot_arrow_head_size
    )

    M_rot_current = [
        Mx_rot[frame],
        My_rot[frame],
        Mz_rot[frame]
    ]

    M_rot_mag_current = np.linalg.norm(M_rot_current)

    if M_rot_mag_current == 0:
        M_rot_mag_current = 1e-12

    M_rot_arrow_head_size = (0.15 / 5 * lim) / M_rot_mag_current * np.sqrt(12)

    M_rot_arrow = ax_3d_rot.quiver(
        0, 0, 0,
        M_rot_current[0], M_rot_current[1], M_rot_current[2],
        color="magenta",
        arrow_length_ratio=M_rot_arrow_head_size
    )

    B_eff_rot_cursor_point.set_data(
        [B_eff_rot[0]],
        [B_eff_rot[1]]
    )
    B_eff_rot_cursor_point.set_3d_properties([B_eff_rot[2]])

    M_rot_cursor_point.set_data(
        [M_rot_current[0]],
        [M_rot_current[1]]
    )
    M_rot_cursor_point.set_3d_properties([M_rot_current[2]])

    M_rot_trace.set_data(
        Mx_rot[:frame + 1],
        My_rot[:frame + 1]
    )
    M_rot_trace.set_3d_properties(Mz_rot[:frame + 1])

    return (
        time_plot_B_eff_x_rot,
        time_plot_B_eff_y_rot,
        time_cursor_point_B_eff_x_rot,
        time_cursor_point_B_eff_y_rot,
        B_eff_rot_arrow,
        M_rot_arrow,
        B_eff_rot_cursor_point,
        M_rot_cursor_point,
        M_rot_trace
    )

animation_rot = FuncAnimation(
    fig=fig_rot,
    func=update_data_rot,
    frames=len(t),
    interval=interval,
    blit=False
)

# =============================================================================
#                           SHARED PAUSE/PLAY BUTTON
# =============================================================================

paused = False

def toggle_animation(event):
    global paused

    if paused:
        animation_lab.event_source.start()
        animation_rot.event_source.start()
        button.label.set_text("Pause")
        paused = False
    else:
        animation_lab.event_source.stop()
        animation_rot.event_source.stop()
        button.label.set_text("Play")
        paused = True

button_ax = fig_lab.add_axes([0.35, 0.05, 0.3, 0.075])
button = Button(button_ax, "Pause")

button.on_clicked(toggle_animation)

plt.show()