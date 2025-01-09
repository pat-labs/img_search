import numpy as np
from matplotlib import pyplot as plt


def barPlot(
    data, labels, title, x_label, y_label, legend_title, legend_data, destiny_path
):
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    colors = plt.cm.BuPu(np.linspace(0, 0.5, len(labels)))
    bars = ax.bar(labels, data, color=colors, width=0.4)

    table = plt.table(
        cellText=legend_data, rowLabels=legend_title, colLabels=labels, loc="bottom"
    )

    # Remove axes splines
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(False)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    # Add x, y gridlines
    ax.grid(True)
    ax.bar_label(bars, fmt="%.4f")

    # Adjust layout to make room for the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Add Plot labels
    ax.set_title(
        title,
        loc="left",
    )
    ax.set_ylabel(y_label)
    plt.xticks([])

    # Adjust the plot to fit the table
    plt.subplots_adjust(left=0.2, bottom=0.1)

    # Save Plot
    plt.savefig(destiny_path)
