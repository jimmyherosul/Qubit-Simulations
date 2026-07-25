# ----------------------------- Import libraries -----------------------------
import numpy as np


def create_signal_plot(ax_tuning, sequence_param, signal_spec, title):
    ax_tuning.set_title(title, fontsize=12, pad=10)                                                                 
    ax_tuning.set_xlim(sequence_param["t_points"][0]*1e9, sequence_param["t_points"][-1]*1e9)
    ax_tuning.set_ylim(-11, 11)
    ax_tuning.set_ylabel("Energy (ueV)", fontsize=10)
    ax_tuning.minorticks_on()
    ax_tuning.grid(True, which="major", linestyle="-", linewidth=0.8)
    ax_tuning.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    signal_function = {}

    for signal in signal_spec:
        key = signal["key"]
        label = signal["label"]

        signal_function[key], = ax_tuning.plot([], [], label=label)

    time_cursor = ax_tuning.axvline(sequence_param["t_points"][0]*1e9, color='r', linestyle='--', linewidth=1.5)  
    signal_cursor, = ax_tuning.plot([], [], 'ro', markersize=4)                                                     
    ax_tuning.legend(loc="upper right")

    tuning_readout = ax_tuning.text(
        1.04, 0.50,
        "",
        transform=ax_tuning.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9)
    )

    return {
        "signal_function": signal_function,
        "time_cursor": time_cursor,
        "signal_cursor": signal_cursor,
        "readout": tuning_readout
    }


def update_signal_plot(frame, sequence_param, plot_objects, signal_spec, frequency_spec):
    t_points_ns = sequence_param["t_points"][:frame+1]*1e9
    t_current_ns = sequence_param["t_points"][frame]*1e9

    current_values = []
    cursor_readout = ["Cursor", f"t = {t_current_ns:.3f} ns"]

    for signal in signal_spec:
        key = signal["key"]
        values_uev = sequence_param[key][:frame + 1] * 1e6
        current_value_uev = sequence_param[key][frame] * 1e6

        plot_objects["signal_function"][key].set_data(t_points_ns, values_uev)

        current_values.append(current_value_uev)
        cursor_readout.append(
            f"{signal['readout_label']} = "
            f"{current_value_uev:.3f} ueV"
        )
    
    plot_objects["time_cursor"].set_xdata([t_current_ns, t_current_ns])
    plot_objects["signal_cursor"].set_data([t_current_ns] * len(current_values), current_values)

    for frequency in frequency_spec:
        frequency_ghz = (sequence_param[frequency["key"]][frame]*1e-9/(2*np.pi))

        cursor_readout.append(
            f"{frequency['readout_label']} = "
            f"{frequency_ghz:.3f} GHz"
        )

    plot_objects["readout"].set_text("\n".join(cursor_readout))