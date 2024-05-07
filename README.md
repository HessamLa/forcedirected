### EXAMPLE

To Embedd a given graph:

```
$ python -m forcedirected embed fd-basic --help 
$ python -m forcedirected embed fd-shell --help
$ python -m forcedirected embed fd-targets --help
```

Such as:

```
python -m forcedirected embed fd-targets --ndim 128 --edgelist ./data/graphs/cora/cora_edgelist.txt --epochs 1000 --name cora --verbosity 2
```

To generate synthetic graph:

```
python -m forcedirected generate --help
python -m forcedirected generate lfr --help 
```

Such as 
```
python -m forcedirected generate lfr -n 1000 # Use LFR benchmark algorhtm to generate graph with 1000 nodes
```



