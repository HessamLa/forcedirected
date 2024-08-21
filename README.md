### To install
It's suggested to install new packages in a virtual environment. But this one is safe if you know what you are doing.
```bash
$ pip install -e .
```

### EXAMPLES

To produce node embeddings of an input graph:

```bash
$ python -m forcedirected embed fdbasic --help 
$ python -m forcedirected embed fdshell --help
$ python -m forcedirected embed fdtargets --help
$ python -m forcedirected embed node2vec --help

```

Such as:

```bash
$ python -m forcedirected embed fdtargets --ndim 128 --edgelist ./data/graphs/cora/cora_edgelist.txt --epochs 1000 --name cora --verbosity 2
```

The string after `embed`, such as `fdtargets` and `fdshell` indicate the embedding method.

To generate synthetic graph:

```bash
$ python -m forcedirected generate --help
$ python -m forcedirected generate lfr --help 
```

Such as:
```bash
$ python -m forcedirected generate lfr -n 1000 # Use LFR benchmark algorhtm to generate graph with 1000 nodes
```

### CODE USAGE

If you want to use the force-directed module in code, you could have a look at the `embed_*` functions in `./forcedirected/embed/fd.py`.

The actualy force-directed class is in `./forcedirected/models`. The file `ForceDirected.py` contains the base class which other classes must derive from. Other files include a class for each method. The main difference between methods are the criteria for picking node pairs for calculating the force between them. For example `model_201_basic` takes all available node pairs and calculates the forces in between with the same coefficients. 

The implementation for [^fdshell,^fdproof] papers is `model_204_shell`. In [^fdproof] the emphasis was more on a mathematical proof.

[^fdshell] Lotfalizadeh, Hamidreza, and Mohammad Al Hasan. "_Force-directed graph embedding with hops distance._" 2023 IEEE International Conference on Big Data (BigData). IEEE, 2023.

[^fdproof] Lotfalizadeh, Hamidreza, and Mohammad Al Hasan. "_Kinematic-Based Force-Directed Graph Embedding._" International Conference on Complex Networks. Cham: Springer Nature Switzerland, 2024.