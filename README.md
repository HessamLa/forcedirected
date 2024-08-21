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

If you want to use the force-directed method in code, you could have a look at the `embed_*` functions in `./forcedirected/embed/fd.py`.
