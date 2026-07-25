def create_coordinate_plot(ax_coordinates, sequence_param, title, show_readout=True):
    ax_coordinates.set_title(title, fontsize=12, pad=10)            
    ax_coordinates.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
    ax_coordinates.set_ylim(-1.1, 1.1)
    ax_coordinates.set_ylabel("Pauli Expectation Values", fontsize=10)
    ax_coordinates.minorticks_on()
    ax_coordinates.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax_coordinates.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    coordinate_function = {}

    coordinate_function["x"], = ax_coordinates.plot([], [], label=r"$\langle\sigma_x\rangle$")
    coordinate_function["y"], = ax_coordinates.plot([], [], label=r"$\langle\sigma_y\rangle$")
    coordinate_function["z"], = ax_coordinates.plot([], [], label=r"$\langle\sigma_z\rangle$")
    coordinate_function["r_mag"], = ax_coordinates.plot([], [], color="black", label=r"$|\mathbf{r}|$")

    time_cursor = ax_coordinates.axvline(sequence_param["t_points"][0]*1e9, color='r', linestyle='--', linewidth=1.5)
    coordinate_cursor, = ax_coordinates.plot([], [], 'ro', markersize=4)
    ax_coordinates.legend(loc="upper right")

    coordinate_readout = None

    if show_readout:
        coordinate_readout = ax_coordinates.text(
            1.04, 0.50,
            "",
            transform=ax_coordinates.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
        )

    return {
        "coordinate_function": coordinate_function,
        "time_cursor": time_cursor,
        "coordinate_cursor": coordinate_cursor,
        "readout": coordinate_readout
    }


def update_coordinate_plot(frame, sequence_param, plot_objects, coordinate_data):
    t_points_ns = sequence_param["t_points"][:frame+1]*1e9
    t_current_ns = sequence_param["t_points"][frame]*1e9

    x = coordinate_data["x"]
    y = coordinate_data["y"]
    z = coordinate_data["z"]
    r_mag = coordinate_data["r_mag"]

    plot_objects["coordinate_function"]["x"].set_data(t_points_ns, x[:frame+1])
    plot_objects["coordinate_function"]["y"].set_data(t_points_ns, y[:frame+1])
    plot_objects["coordinate_function"]["z"].set_data(t_points_ns, z[:frame+1])
    plot_objects["coordinate_function"]["r_mag"].set_data(t_points_ns, r_mag[:frame+1])

    plot_objects["time_cursor"].set_xdata([t_current_ns, t_current_ns])
    plot_objects["coordinate_cursor"].set_data(
        [t_current_ns, t_current_ns, t_current_ns, t_current_ns],
        [x[frame], y[frame], z[frame], r_mag[frame]]
    )

    if plot_objects["readout"] is not None:
        plot_objects["readout"].set_text(
            "Cursor\n"
            f"t = {t_current_ns:.3f} ns\n"
            f"<sigma_x> = {x[frame]:.3f}\n"
            f"<sigma_y> = {y[frame]:.3f}\n"
            f"<sigma_z> = {z[frame]:.3f}\n"
            f"|r| = {r_mag[frame]:.3f}"
        )