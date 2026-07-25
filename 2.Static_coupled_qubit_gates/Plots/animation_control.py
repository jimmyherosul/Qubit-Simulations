from matplotlib.widgets import Button

class animation_control:
    def __init__(self, anim, animation_figure, button_figure):
        self.anim = anim
        self.animation_figure = animation_figure
        self.paused = True
        self.animation_finished = False

        button_ax = button_figure.add_axes([0.45, 0.03, 0.1, 0.05])
        self.button = Button(button_ax, "Play")
        self.button.on_clicked(self.toggle_animation)

        self.first_draw_cid = self.animation_figure.canvas.mpl_connect(
            "draw_event", 
            self.pause_animation_on_first_draw
        )

    # Pause/Play button
    def toggle_animation(self, event):
        if self.paused:
            if self.animation_finished:
                self.anim.frame_seq = self.anim.new_frame_seq()
                self.animation_finished = False

            if self.anim.event_source is not None:
                self.anim.event_source.start()

            self.button.label.set_text("Pause")
            self.paused = False

        else:
            if self.anim.event_source is not None:
                self.anim.event_source.stop()

            self.button.label.set_text("Play")
            self.paused = True


    # Stop the animation immediately after the first draw event
    def pause_animation_on_first_draw(self, event):
        if event.canvas != self.animation_figure.canvas:
            return
        
        if self.anim.event_source is not None:
            self.anim.event_source.stop()

        self.button.label.set_text("Play")
        self.paused = True
        self.animation_finished = False

        self.animation_figure.canvas.mpl_disconnect(self.first_draw_cid)


    def finish_animation(self):
        if self.anim.event_source is not None:
            self.anim.event_source.stop()

        self.button.label.set_text("Play")
        self.paused = True
        self.animation_finished = True
