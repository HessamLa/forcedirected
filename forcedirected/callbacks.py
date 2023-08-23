
import os
import sys
import pickle
import pandas as pd
import logging
import torch
from .utilityclasses import Callback_Base
from .utilities import ReportLog
from .utilities import optimize_batch_count

@torch.no_grad()
def make_hops_stats_old(Z, hops, maxhops):
    print("make_hops_stats")
    # print("make_hops_stats", Z.device, f"block_size:{block_size}", f"ndim:{Z.size(1)}", f"block bytesize:{Z.element_size()*block_size*Z.size(1):_d}")
    maxhops = int(maxhops)
    sum_N=torch.zeros(maxhops+2).to(Z.device)
    sum_N2=torch.zeros(maxhops+2).to(Z.device)

    i,j = torch.triu_indices(Z.size(0), Z.size(0), offset=1) # offset 1 to exclude self distance
    """Z is node embedding, torch tensor"""
    # empty cuda cache
    # gc.collect()
    # torch.cuda.empty_cache()
    # get available cuda memory
    # available_memory = torch.cuda.memory_cached(Z.device) - torch.cuda.memory_allocated(Z.device)
    available_memory = torch.cuda.get_device_properties(Z.device).total_memory//2
    block_size = int(available_memory) // (Z.element_size()*Z.size(1)) # 10GB
    print(f"  available_memory {available_memory/2**30:.2f} GB")
    print(f"  block_size {block_size} elements. Block count {len(i)//block_size+1}")
    print(f"  block_size {block_size*Z.element_size()*Z.size(1)/2**30:.2f} GB")
        
    while block_size>0:
        try:
            for b in range(0, len(i), block_size):
                i_block = i[b:b+block_size]
                j_block = j[b:b+block_size]
                # print(f"block={b}", f"i_block:{i_block[0]}-{i_block[-1]}", f"j_block:{j_block[0]}-{j_block[-1]}")
                hops_block = hops[i_block, j_block]
                N_block = torch.norm(Z[j_block] - Z[i_block], dim=-1)

                for h in range(1, maxhops+1):
                    hmask = hops_block==h
                    sum_N.add_(N_block[hmask].sum())
                    sum_N2.add_((N_block[hmask]**2).sum())
                    
                    # sum_N[h] += N_block[hmask].sum()
                    # sum_N2[h] += (N_block[hmask]**2).sum()
                # disconnected components
                hmask = hops_block>maxhops 
                sum_N[maxhops+1] += N_block[hmask].sum()
                sum_N2[maxhops+1] += (N_block[hmask]**2).sum()
                del hops_block, N_block
            break
        except RuntimeError as e:
            if('out of memory' in str(e)):
                block_size = int(block_size/2)
                print(f"  block_size {block_size} elements. Block count {len(i)//block_size+1}")
                print(f"  block_size {block_size*Z.element_size()*Z.size(1)/2**30:.2f} GB")
            
    print("make_hops_stats done")
    s={}
    s.update({f"hops{h}_mean": (sum_N[h]/(hops==h).sum()).item() for h in range(1, maxhops+1)})
    s.update({f"hops{h}_std":  (sum_N2[h]/(hops==h).sum() - (sum_N[h]/(hops==h).sum())**2).sqrt().item() for h in range(1, maxhops+1)})
    # disconnected components
    if((hops>maxhops).sum()>0):
        s[f"hops{maxhops}_mean"] = (sum_N[maxhops]/(hops>maxhops).sum()).item()
        s[f"hops{maxhops}_std"]  = (sum_N2[maxhops]/(hops>maxhops).sum() - (sum_N[maxhops]/(hops>maxhops).sum())**2).sqrt().item()
    return s
class batch_optimizer:
    def __init__(self, func) -> None:
        self.func = func
        self.memo = {}
        pass
    def __call__(self, *args, **kwargs):
        if('batch_count' in kwargs):
            return
        
@torch.no_grad()
@optimize_batch_count()
def make_hops_stats(Z, hops, maxhops, batch_count=1, *args, **kwargs):
    print(f"make_hops_stats-optimized started with batch_count={batch_count}")
    maxhops = int(maxhops)    
    i,j = torch.triu_indices(Z.size(0), Z.size(0), offset=1) 
    hops_triu = hops[i,j]

    sum_N = torch.zeros(maxhops+2, device=Z.device)
    sum_N2 = torch.zeros(maxhops+2, device=Z.device)
    
    block_size = int(len(i)/batch_count + 0.5)
    for b in range(0, len(i), block_size):
        i_block = i[b:b+block_size] 
        j_block = j[b:b+block_size]            
        Z_i = Z[i_block]
        Z_j = Z[j_block]            
        # hops_block = hops[i_block, j_block]             
        hops_block = hops_triu[b:b+block_size]
        N2_block = torch.sum((Z_i - Z_j)**2, dim=-1) # squared norms                    
        for h in range(1, maxhops+1):                    
            hmask = (hops_block == h)                        
            sum_N[h].add_(N2_block[hmask].sqrt().sum()) # sqrt of sum
            sum_N2[h].add_(N2_block[hmask].sum())             
        # disconnected components
        hmask = (hops_block > maxhops)  
        sum_N[maxhops+1].add_(N2_block[hmask].sqrt().sum())
        sum_N2[maxhops+1].add_(N2_block[hmask].sum())    
    # print(f"  block_size {block_size*Z.element_size()*Z.size(1)/2**30:.2f} GB. Block count {batch_count}")
    # print("make_hops_stats-optimized done")

    s = {}
    for h in range(1, maxhops+1):
        n = (hops_triu==h).sum()
        smean = sum_N[h]/n
        s.update({f"hops{h}_mean": smean.item()})
        if(n>1):
            sstd = (sum_N2[h]/n - smean**2).sqrt()
            s.update({f"hops{h}_std": sstd.item()})
        else:
            s.update({f"hops{h}_std": 0.0})
    # disconnected components
    n = (hops_triu>maxhops).sum()
    smean = sum_N[h]/n
    s.update({f"hopsinf_mean": smean.item()})
    if(n>1):
        sstd = (sum_N2[h]/n - smean**2).sqrt()
        s.update({f"hopsinf_std": sstd.item()})
    else:
        s.update({f"hopsinf_std": 0.0})

    return s
    
def make_force_stats(model):
    summary_stats = lambda x: (torch.sum(x).item(), torch.mean(x).item(), torch.std(x).item())
    s={}
    fsum=torch.zeros_like(model.forces[0])
    for F in model.forces:
        print(F.tag)
        # s[f'{f.name}-sum'], s[f'{f.name}-mean'], s[f'{f.name}-std'] = summary_stats( torch.norm(forces[f], dim=1) )
        # s[f'{f.name}-sum-w'], s[f'{f.name}-mean-w'], s[f'{f.name}-std-w'] = summary_stats( torch.norm(forces[f], dim=1) )*model.degrees
        # fall += f

    # s['f-sum'] = torch.norm(fsum, dim=-1).sum().item()
    # s['f-sum-w'] = torch.norm(fsum*model.degrees[:, None], dim=-1).sum().item()
    # s['f-sum-mag'] = torch.norm(fsum, dim=1).sum().item()
    return s

def make_stats_log(model, epoch):
    logstr = ''
    s = {'epoch': epoch}
    s.update(make_hops_stats(model.Z, model.hops, model.maxhops))

    make_force_stats(model)
    summary_stats = lambda x: (torch.sum(x).item(), torch.mean(x).item(), torch.std(x).item())

    # attractive forces
    Fa = model.fmodel_attr.F
    s['fa-sum'],  s['fa-mean'],  s['fa-std']  = summary_stats( torch.norm(Fa, dim=1) )
    # weighted attractive forces
    s['wfa-sum'], s['wfa-mean'], s['wfa-std'] = summary_stats( torch.norm(Fa, dim=1)*model.degrees )

    # repulsive forces
    Fr = model.fmodel_repl.F
    s['fr-sum'],  s['fr-mean'],  s['fr-std']  = summary_stats( torch.norm(Fr, dim=1) )
    # weighted repulsive forces
    s['wfr-sum'], s['wfr-mean'], s['wfr-std'] = summary_stats( torch.norm(Fr, dim=1)*model.degrees )
    
    
    # sum of all forces, expected to converge to 0
    s['f-all'] = torch.norm( torch.sum(Fa+Fr, dim=0) ) 
    
    # sum of norm/magnitude of all forces, expected to converge
    s['f-all-sum'] = s['fa-sum'] + s['fr-sum']

    # sum of all weighted forces, expected to converge to 0
    s['wf-all'] = torch.norm( torch.sum( (Fa+Fr)*model.degrees[:, None], dim=0 ) )    

    # sum of norm/magnitude of all weighted forces, expected to converge
    s['wf-all-sum'] = s['wfa-sum'] + s['wfr-sum']

    # relocations
    s['relocs-sum'], s['relocs-mean'], s['relocs-std'] = summary_stats( torch.norm(model.dZ, dim=1) )
    
    # weighted relocations
    s['wrelocs-sum'], s['wrelocs-mean'], s['wrelocs-std'] = summary_stats( torch.norm(model.dZ, dim=1)*model.degrees )
    
    # convert all torch.Tensor elements to regular python numbers
    for k,v in s.items():
        if(type(v) is torch.Tensor):
            s[k] = v.item()

    logstr = f"attr:{s['fa-sum']:<9.3f}({s['fa-mean']:.3f})  "
    logstr+= f"repl:{s['fr-sum']:<9.3f}({s['fr-mean']:.3f})  "
    logstr+= f"wattr:{s['wfa-sum']:<9.3f}({s['wfa-mean']:.3f})  "
    logstr+= f"wrepl:{s['wfr-sum']:<9.3f}({s['wfr-mean']:.3f})  "
    logstr+= f"sum-all:{s['f-all']:<9.3f}  "
    logstr+= f"relocs:{s['relocs-sum']:<9.3f}({s['relocs-mean']:.3f})  "
    logstr+= f"weighted-relocs:{s['wrelocs-sum']:<9.3f}({s['wrelocs-mean']:.3f})  "

    return s, logstr

class StatsLog (Callback_Base):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.args=kwargs['args']
        self.emb_filepath_tmp = f"{self.args.outputdir}/{self.args.outputfilename}.tmp" # the path to store the latest embedding
        self.emb_filepath = f"{self.args.outputdir}/{self.args.outputfilename}" # the path to store the final embedding
        os.system(f"rm -f {self.emb_filepath_tmp} {self.emb_filepath}") # remove the old embedding files

        self.hist_filepath = f"{self.args.outputdir}/{self.args.historyfilename}" # the path to APPEND the latest embedding
        self.stats_filepath = f"{self.args.outputdir}/{self.args.statsfilename}" # the path to save the latest stats
        self.save_history_every = self.args.save_history_every
        self.save_stats_every = self.args.save_stats_every
        self.statsdf = pd.DataFrame()
        # self.logger = ReportLog(self.args.logfilepath)
        
        # make he stat logger #######################
        self.statlog = logging.getLogger('StatsLog')
        self.statlog.setLevel(logging.INFO)

        # Add a file handler to write logs to a file
        file_handler = logging.FileHandler(self.args.logfilepath, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s: %(message)s'))
        self.statlog.addHandler(file_handler)

        # Add a console handler to print logs to the console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        # console_handler.setFormatter(logging.Formatter('> %(message)s'))
        self.statlog.addHandler(console_handler)
        ############################################
    

    def save_embeddings(self, fd_model, **kwargs):
        emb = fd_model.get_embeddings()
        # save embeddings as pandas df
        df = pd.DataFrame(emb, index=fd_model.Gx.nodes())
        df.to_pickle(self.emb_filepath_tmp) # save to temporary file. will be renamed in train done
    
    def save_history(self, fd_model, **kwargs):
        emb = fd_model.get_embeddings()
        # save embeddings history
        with open(self.hist_filepath, "ab") as f: # append embeddings
            pickle.dump(emb, f)

    def update_stats(self, fd_model, epoch, **kwargs):
        # make stats and logs
        stats, statstr = make_stats_log(fd_model, epoch)

        # make the stats as dataframe
        if(len(self.statsdf) == 0):  # new dataframe
            self.statsdf = pd.DataFrame(columns=list(stats.keys()))    
        
        stats = pd.DataFrame(stats, index=[0])

        self.statsdf = pd.concat([self.statsdf, stats], ignore_index=True)

        # Save DataFrame to a CSV file
        # temp_filename = self.stats_filepath+'_tmp'
        self.statsdf.to_csv(self.stats_filepath, index=False)
        # Rename the temporary file to the final filename
        # os.rename(temp_filename, self.stats_filepath)
        logstr = f"Epoch {epoch+1}/{kwargs['epochs']}  ({kwargs['batch_count']} batches) | {statstr}"
        self.statlog.info(logstr) # FIX THIS

    def on_epoch_begin(self, fd_model, epoch, epochs, **kwargs):
        self.statlog.debug(f'Epoch {epoch+1}/{epochs}')
        # return super().on_batch_begin(fd_model, epoch, **kwargs)

    def on_epoch_end(self, fd_model, epoch, **kwargs):
        self.statlog.debug(f"   Batch size: {kwargs['batch_size']}")
        self.save_embeddings(fd_model, **kwargs)
        if(epoch % self.save_stats_every == 0):
            self.update_stats(fd_model, epoch, **kwargs)
        if(epoch % self.save_history_every == 0):
            self.save_history(fd_model, **kwargs)
            
    def on_batch_end(self, fd_model, batch, **kwargs):
        self.statlog.debug(f"   Batch {batch}/{kwargs['batch_count']}")
        # return super().on_batch_begin(fd_model, batch, **kwargs)

    def on_train_end(self, fd_model, epochs, **kwargs):
        print("on_train_end() ---+-----+--- train ended here")
        kwargs['epochs']=epochs
        self.statlog.debug("Final save")
        self.save_embeddings(fd_model, **kwargs)
        self.update_stats(fd_model, **kwargs)
        self.save_history(fd_model, **kwargs)

        # rename the temporary embedding file
        # os.rename(self.emb_filepath_tmp, self.emb_filepath)
        # I don't trust os.rename. So I use system command instead
        os.system(f"mv {self.emb_filepath_tmp} {self.emb_filepath}")
        
