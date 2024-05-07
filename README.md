### EXAMPLE
python3 -m forcedirected --fdversion 121  --outputdir_root ./embeddings-tmp-240106-1502 --dataset_name corafull --ndim 128 --edgelist /N/u/hessamla/BigRed200/gnn/datasets/corafull/corafull_edgelist.txt --nodelist /N/u/hessamla/BigRed200/gnn/datasets/corafull/corafull_nodes.txt --epochs 5000 --save-stats-every 1

    --fdversion: Version of the model as found in ./models/model_<version>.py
    --dataset_name: Name of the dataset, used for naming output files
    --edgelist: Path to the edgelist file
    --nodelist: (optional) Path to the nodelist file. This files can be used to maintain the order of nodes as read by NetworkX.
    --outputdir_root: 
    --ndim: Number of dimensions
    --epochs: Number of iterations.
    --save-stats-every: The period for saving stats.