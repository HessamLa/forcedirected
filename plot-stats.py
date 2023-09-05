# %%
# Embeddings-tmp is the directory where the embeddings and
# statistics are stored.
# The directory structure is embeddings-tmp/{method}/{dataset}/
# In each dataset directory access stats.csv and plot it using the
# following code

# %%

import os
from forcedirected.utilities.loaders import load_stats
import pandas as pd
import matplotlib.pyplot as plt
from forcedirected.utilities.RecursiveNamespace import RecursiveNamespace as rn



def plot_stats(method, dataset, statspath, ax=None, epoch_range=None,
                save_to=None, figtitle=None, from_time=None, to_time=None):
    df = load_stats(statspath)
    print(f'loaded dataframe from {statspath}')
    # TEMPORARY HACK
    stdcols = [c for c in df.columns if c.endswith('_std') and
               c.startswith('hops') and not c.startswith('hopsinf')]
    meancols = [c for c in df.columns if c.endswith('_mean') and
                c.startswith('hops') and not c.startswith('hopsinf')]

    plt.close("all")
    if (from_time is not None):
        df = df[df.epoch>=from_time]
    if (to_time is not None):
        df = df[df.epoch<=to_time]
    if (epoch_range is not None):
        df = df.iloc[epoch_range[0]:epoch_range[1]]

    df.index = df.epoch
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        10, 5), gridspec_kw={'width_ratios': [1, 1]})
    # if(figtitle is None):
    #     fig.suptitle(f'Pairwise distance (Dataset:{dataset}  Method:{method})')
    # elif(figtitle != ''):
    #     fig.suptitle(figtitle)
    ax1.set_xlabel('epoch')
    ax2.set_xlabel('epoch')

    df[stdcols].plot(title='std', ax=ax1, legend=False)

    # ax = df[meancols].iloc[-400:].plot(title=f'dataset: {profile.dataset}, method: {profile.method}')
    ax2 = df[meancols].plot(title='mean', ax=ax2)

    # Retrieve the line colors used in the plot
    line_colors = [line.get_color() for line in ax2.get_lines()]

    # Define the legend with colored circles matching the line colors
    legend_labels = df[meancols].columns.str.strip('_mean').str.strip('_std')
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8)
                       for color in line_colors]

    # Place the legend right to the figure, out of the axis
    ax2.legend(reversed(legend_elements), reversed(legend_labels),
               loc='center left', bbox_to_anchor=(1, 0.5))
    # Enable gridlines on major ticks
    ax2.grid(True)
    ax2.set_ylim(bottom=0.0)
    # Adjust the layout to accommodate the legend
    # plt.subplots_adjust(right=0.75)
    # SAVE PLOT
    if(save_to):
        save_to = save_to or f'images/pairdist-{dataset}-{method}.png'
        print(f'saving to {save_to}')
        plt.savefig(save_to, bbox_inches='tight')
    # Show the plot
    plt.show()

path='/N/u/hessamla/BigRed200/gnn/forcedirected/embeddings-gpu-bigred200/forcedirected_v0004_128d/corafull/stats.csv'
path='/N/u/hessamla/BigRed200/gnn/forcedirected/embeddings-gpu-debug-bigred200/forcedirected_v0006_64d/ego-facebook/stats.csv'
path='/N/u/hessamla/BigRed200/gnn/forcedirected/embeddings-tmp/forcedirected_v0106_128d/ego-facebook/stats.csv'

dataset='ego-facebook'
dataset='cora'
dataset='citeseer'
dataset='pubmed'
method='forcedirected_v0104_128d'
path=f'/N/u/hessamla/BigRed200/gnn/forcedirected/embeddings-tmp/{method}/{dataset}/stats.csv'
saveto=f'images/pairdist-{dataset}-{method}.pdf'
plot_stats(method, dataset, path, save_to=saveto, from_time=1)
# %%
# Directory where the embeddings and statistics are stored
base_dir = 'embeddings-tmp'
base_dir = 'embeddings-gpu-bigred200'
base_dir = 'embeddings-gpu-carbonate'

# Iterate over subfolders
paths=[]
for method in os.listdir(base_dir):
    #     if('v0003' not in method): continue
    method_dir = os.path.join(base_dir, method)

    if os.path.isdir(method_dir):
        for dataset in os.listdir(method_dir):
            dataset_dir = os.path.join(method_dir, dataset)
            stats_path = os.path.join(dataset_dir, 'stats.csv')
            
            paths.append(rn(method=method, dataset=dataset, stats_path=stats_path))

for p in paths:
    # if('cora' not in p.stats_path or 'corafull' in p.stats_path): continue
    if('wiki' not in p.stats_path or 'corafull' in p.stats_path): continue
    if('v0005' not in p.stats_path): continue
    if('_128d' not in p.stats_path): continue
    print(p)
    if os.path.isfile(p.stats_path):
        plot_stats(p.method, p.dataset, p.stats_path)

# %%
# plot_stats('v4', 'cora test', '/N/u/hessamla/BigRed200/gnn/forcedirected/testdir/stats.csv')

# %%
import matplotlib.pyplot as plt
import numpy as np

n = 3  # Red shades
k = 5  # Green shades
n2 = n + 2  # Blue shades

heights = np.random.randint(1, 10, size=n+k+n2)
# Generate shades of Red (light to dark)
colors1 = plt.cm.Reds(np.linspace(0.4, 0.8, n))
print(colors1)
# Generate shades of Green (dark to light)
colors2 = plt.cm.Greens(np.linspace(0.2, 0.8, k))[::-1]
print(colors2)
# Generate shades of Blue (light to dark)
colors3 = plt.cm.OrRd(np.linspace(0.2, 0.8, n2))
print(colors3)
# Combine the color arrays using a list

colors_set = [colors1, colors2, colors3]
# Create patterns for each group of bar colors
patterns = ['/', '\\', 'x', '+', 'o']
patterns = [patterns[i]*len(c) for i, c in enumerate(colors_set)]
patterns = ''.join(patterns)
print(patterns)


# Create vertical bar plots with patterns in one axis
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate the width of each bar group
bar_width = 0.6 / len([c for c in colors_set])

gap=0.1
x_pos = 0
i=0
for colors in colors_set:
    x_pos += gap
    for c in colors:
        # Add pattern to the bars
        x_pos += bar_width
        ax.bar(x_pos, 1, color=c, 
               edgecolor='black', 
               hatch=patterns[i], 
               width=bar_width, orientation='vertical')
        i+=1

ax.set_xticks([i * bar_width + bar_width / 2 for i in range(len(colors))])
ax.set_xticklabels([f'Color {i+1}' for i in range(len(colors))], rotation=45)
ax.set_yticks([])
ax.set_title('Color Bar Plot with Patterns')
plt.show()