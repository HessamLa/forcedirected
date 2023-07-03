def make_embedding_stats(Nt, hops, maxhops):
    """Nt is pairwise euclidean distance, Numpy
    hops is pairwise hop distance, Numpy
    """
    ht = hops[hops<=maxhops]
    
    stats = {}
    for l in range(1, int(ht.max()+1)):
        mask = hops==l
        if(len(Nt[mask])>0):
            stats[f"hops{l}_mean"]=Nt[mask].mean()
            stats[f"hops{l}_std"]=Nt[mask].std()
        # print(f"{l:3d} {Nt[mask].mean():10.3f} {Nt[mask].std():10.3f} {len(Nt[mask])/2:8.0f}")
    
    # disconnected components
    mask = hops>maxhops
    if(len(Nt[mask])>0):
        stats[f"hopsinf_mean"]=Nt[mask].mean()
        stats[f"hopsinf_std"]=Nt[mask].std()
    return stats
    