def create_energy_plot(ax_energy, results_in_parallel):
    ax_energy.set_xlabel(r"$\Delta$ (ueV)", fontsize=11)
    ax_energy.set_ylabel(r"$E-\overline{E}$ (ueV)", fontsize=11, labelpad=2)
    ax_energy.tick_params(axis='both', labelsize=9)
    ax_energy.minorticks_on()
    ax_energy.grid(True, which="major", linestyle="-", linewidth=0.6)
    ax_energy.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.7)

    unperturbed_energy = {}

    unperturbed_energy["E_plus"], = ax_energy.plot([], [], color="black", linestyle="--", label=r"$|V|=0$ ueV")
    unperturbed_energy["E_minus"], = ax_energy.plot([], [], color="black", linestyle="--")

    perturbed_energy = []

    for test_sim in results_in_parallel:
        Delta = test_sim["Delta"]
        V_mag = test_sim["V_mag"]
        V_mag_ueV = int(round(V_mag*1e6))
        curve_colour = test_sim["curve_colour"]
        
        energy_function = {}

        energy_function["E_plus"], = ax_energy.plot([], [], color=curve_colour, label=fr"$|V|={V_mag_ueV}$ $\mu$eV")
        energy_function["E_minus"], = ax_energy.plot([], [], color=curve_colour)
        energy_function["E_plus_current"], = ax_energy.plot([], [], 'o', color=curve_colour, markersize=4, zorder=5)
        energy_function["E_minus_current"], = ax_energy.plot([], [], 'o', color=curve_colour, markersize=4, zorder=5)

        delta_current = ax_energy.axvline(Delta*1e6, color="red", linestyle=":", linewidth=1.0, zorder=1)

        perturbed_energy.append(energy_function)

    ax_energy.legend(loc="best", fontsize=6)

    return {
        "unperturbed_energy": unperturbed_energy,
        "perturbed_energy": perturbed_energy
    }