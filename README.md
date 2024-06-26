### To install
It's suggested to install new packages in a virtual environment. But this one is safe if you know what you are doing.
```bash
$ pip install -e .
```

### EXAMPLE

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

To generate synthetic graph:

```bash
$ python -m forcedirected generate --help
$ python -m forcedirected generate lfr --help 
```

Such as:
```bash
$ python -m forcedirected generate lfr -n 1000 # Use LFR benchmark algorhtm to generate graph with 1000 nodes
```



