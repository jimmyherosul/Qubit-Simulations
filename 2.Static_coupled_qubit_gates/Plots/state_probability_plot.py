import numpy as np


def create_probability_plot(ax_probabilities, sequence_param, title, show_readout=True):
    ax_probabilities.set_title(title, pad=10)              
    ax_probabilities.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
    ax_probabilities.set_ylim(0, 1.1)
    ax_probabilities.set_xlabel("Time (ns)", fontsize=10)
    ax_probabilities.set_ylabel("Probability", fontsize=10)
    ax_probabilities.minorticks_on()
    ax_probabilities.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax_probabilities.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    probability_function = {}

    probability_function["p0"], = ax_probabilities.plot([], [], label=r"$P_0(t)$")
    probability_function["p1"], = ax_probabilities.plot([], [], label=r"$P_1(t)$")
    probability_function["p_total"], = ax_probabilities.plot([], [], color="black", label=r"$P_{\mathrm{total}}(t)$")

    time_cursor = ax_probabilities.axvline(sequence_param["t_points"][0]*1e9, color='r', linestyle='--', linewidth=1.5)
    probability_cursor, = ax_probabilities.plot([], [], 'ro', markersize=4)
    ax_probabilities.legend(loc="upper right")

    probability_readout = None

    if show_readout:
        probability_readout = ax_probabilities.text(
            1.04, 0.50,
            "",
            transform=ax_probabilities.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
        )

    return {
        "probability_function": probability_function,
        "time_cursor": time_cursor,
        "probability_cursor": probability_cursor,
        "readout": probability_readout
    } 


def update_probability_plot(frame, sequence_param, plot_objects, probability_data, rabi_frequency, rabi_frequency_label):
    t_points_ns = sequence_param["t_points"][:frame+1]*1e9
    t_current_ns = sequence_param["t_points"][frame]*1e9

    p0 = probability_data["p0"]
    p1 = probability_data["p1"]
    p_total = probability_data["p_total"]

    plot_objects["probability_function"]["p0"].set_data(t_points_ns, p0[:frame+1])
    plot_objects["probability_function"]["p1"].set_data(t_points_ns, p1[:frame+1])
    plot_objects["probability_function"]["p_total"].set_data(t_points_ns, p_total[:frame+1])

    plot_objects["time_cursor"].set_xdata([t_current_ns, t_current_ns])
    plot_objects["probability_cursor"].set_data(
        [t_current_ns, t_current_ns, t_current_ns],
        [p0[frame], p1[frame], p_total[frame]]
    )

    rabi_freq_ghz = (rabi_frequency[frame]*1e-9/(2*np.pi))

    if plot_objects["readout"] is not None:
        plot_objects["readout"].set_text(
            "Cursor\n"
            f"t = {t_current_ns:.3f} ns\n"
            f"P0 = {p0[frame]:.3f}\n"
            f"P1 = {p1[frame]:.3f}\n"
            f"P_total = {p_total[frame]:.3f}\n"
            f"{rabi_frequency_label} = {rabi_freq_ghz:.3f} GHz"
        )
