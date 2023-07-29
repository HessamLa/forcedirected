import random

def batchify(idxlist:list, batch_size:int, shuffle=False):
    if(shuffle):
        idxlist = idxlist.copy()
        random.shuffle(idxlist)
    if(len(idxlist) <= batch_size):
        yield idxlist
        return
    for i in range(0, len(idxlist), batch_size):
        yield idxlist[i:i+batch_size]
