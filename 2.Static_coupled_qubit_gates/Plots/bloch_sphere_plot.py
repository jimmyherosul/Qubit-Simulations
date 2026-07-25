from qutip import Bloch

def create_bloch_sphere(figure, subplot_position, title=None):
    ax_bloch = figure.add_subplot(subplot_position, projection='3d')
    b = Bloch(axes = ax_bloch)
    b.font_size = 14

    return {
        "ax_bloch": ax_bloch,
        "bloch_sphere": b,
        "title": title
    }

def update_bloch_sphere(frame, bloch_view, qubit_state, coordinate_data, point_colour, vector_colour):
    x = coordinate_data["x"]
    y = coordinate_data["y"]
    z = coordinate_data["z"]

    ax_bloch = bloch_view["ax_bloch"]
    bloch = bloch_view["bloch_sphere"]
    title = bloch_view["title"]    

    bloch.clear()
    bloch.point_color = [point_colour]
    bloch.vector_color = [vector_colour]

    bloch.add_states(qubit_state)
    bloch.add_points([x[frame], y[frame], z[frame]], 's')
    bloch.add_points([x[:frame+1], y[:frame+1], z[:frame+1]], 'l')
    bloch.make_sphere()

    if title is not None:
        ax_bloch.set_title(title, fontsize=12, pad=10)