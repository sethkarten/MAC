try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Qt5agg")
except ImportError:
    matplotlib = None
    plt = None


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig, self.ax = plt.subplots(figsize=(21, 15))

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)
        self.imshow_obj = None
        self.ax.axis('off')

        # Flag indicating the window was closed
        self.closed = False

        # noinspection PyUnusedLocal
        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)

    def clear(self):
        self.ax.clear()

    def set_lim(self, x_values, y_values, scaling_factor=0.2):
        if x_values is not None:
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * scaling_factor
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
        if y_values is not None:
            y_max = max(y_values)
            y_min = min(y_values)
            y_margin = (y_max - y_min) * scaling_factor
            self.ax.set_ylim(y_max + y_margin, y_min - y_margin, )

    def imshow(self, img, text=''):
        self.ax.set_title(text)
        self.ax.imshow(img, interpolation='none', cmap='Pastel1')
        self.fig.canvas.draw()
        self.fig.canvas.start_event_loop(0.001)

    def show(self, left_offset=-3840):
        plt.ion()
        figManager = plt.get_current_fig_manager()
        figManager.window.move(left_offset, 0)
        figManager.window.showMaximized()
        plt.tight_layout(pad=2)
        plt.show()

    @staticmethod
    def close():
        """
        Close the window
        """

        plt.close()
