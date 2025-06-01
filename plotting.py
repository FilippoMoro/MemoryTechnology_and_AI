import numpy as np
import matplotlib.pyplot as plt

# Plotting function for the 
def plot_fitness_table( fitness_coefs, memory_table, nn_table, plot_percent=True,
                        xlabel_size = 15, ylabel_size = 15, quantile_col = 0.25,
                        path_save = '/content/drive/MyDrive/POSTDOC/Projects/Conferences/2025_ISMC_Sofia/Figures',
                        filename = 'fitness_table',
                        title = None,
                        no_ylabel = False,
                        cmap = 'Greens',
                        cbar_show = False,
                        vminmax = [0, 100]
                        ):

  if plot_percent: fitness_coefs = 100 * fitness_coefs

  fig, ax = plt.subplots(figsize=(8, 8))

  len_memory = len(nn_table.values)
  len_nn = len(memory_table.values)

  # Plot the image
  img = ax.imshow(fitness_coefs, cmap=cmap, aspect=(len_memory / len_nn),
                         vmin=vminmax[0], vmax=vminmax[1])

  # Set ticks and labels
  ax.set_xticks(np.arange(len_memory))
  ax.set_xticklabels(['{}'.format(v[0]) for v in nn_table.values], rotation=45, ha='right', size=xlabel_size)

  if not no_ylabel:
    ax.set_yticks(np.arange(len_nn))
    ax.set_yticklabels([v[0] for v in memory_table.values], size=ylabel_size)
  else:
    ax.set_yticks([])

  if title is not None:
    ax.set_title(title, size=15)

  # Colorbar with matched height
  if cbar_show:
      cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
      cbar.ax.tick_params(labelsize=10)

  # Annotate each cell with the value
  # q25 = np.quantile(fitness_coefs, quantile_col)
  q25 = quantile_col
  for i in range(len_nn):
      for j in range(len_memory):
          val = fitness_coefs[i, j]
          if not plot_percent:
            color = 'white' if val > q25 else 'black'
            ax.text(j, i, f"{val:.2f}%", ha='center', va='center', size=15, color=color)
          else:
            color = 'white' if val > q25 else 'black'
            ax.text(j, i, f"{val:.0f}", ha='center', va='center', size=15, color=color)

  # Add grid lines
  ax.set_xticks(np.arange(len_memory + 1) - 0.5, minor=True)
  ax.set_yticks(np.arange(len_nn + 1) - 0.5, minor=True)
  ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
  ax.tick_params(which="minor", bottom=False, left=False)

  plt.tight_layout()
  plt.show()

#   fig.savefig( path_save + '/' + filename + '.pdf', dpi=300, transparent=True )

  return None


def plot_weights( weights, color='k', alpha = 0.25, 
                 path_save = '/content/drive/MyDrive/POSTDOC/Projects/Conferences/2025_ISMC_Sofia/Figures',
                 filename = 'Weights',
                 plot_values = False ):
    fig, ax = plt.subplots(figsize=(2.5, 1))
    ax.bar( np.arange(0, len(weights)), weights, color=color, edgecolor='k', alpha = alpha )
    if plot_values:
        [ax.text( i-0.35, w+0.01, f'{np.round(100*w,0):.0f}' ) for i, w in enumerate(weights) ]
    ax.set_xticks( np.arange(0, len(weights)), labels = [f'w{i+1}' for i in range(len(weights))] )
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # fig.savefig( path_save + '/' + filename + '.pdf', dpi=300, transparent=True )