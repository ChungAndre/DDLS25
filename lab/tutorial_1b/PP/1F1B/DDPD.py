import torch.distributed as dist
import torch
import os
from sys import argv
from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories
from simplellm.losses import causalLLMLoss
from torch.optim import Adam

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
world_size = 6 
dist.init_process_group("gloo", rank=rank, world_size=world_size)
num_group = 2

# Create groups 
group_0 = dist.new_group(ranks=[0, 1, 2])
group_1 = dist.new_group(ranks=[3, 4, 5])

# Assign pipeline, local rank and group
pipeline_id = rank // 3
local_rank = rank % 3
pipeline_group = group_0 if pipeline_id == 0 else group_1
print(f"Process {rank} initialized in pipeline {pipeline_id}, local rank: {local_rank}")

dmodel = 288
num_heads = 6
n_layers = 3 
seq_l = 256
batch_size = 6
device = "cpu"
num_microbatches = 6
microbatch_size = 1


# Assign models
if local_rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                            device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)
elif local_rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads, 
                        device=device, n_layers=n_layers, ctx_size=seq_l)
elif local_rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads, 
                            device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)

for itr in range(5_000):
    optim.zero_grad()

    # Forward pass
    for microbatch in range(num_microbatches):
        print(f"Pipeline {pipeline_id} Rank {rank} processing microbatch {microbatch}")
        if local_rank == 0:
            try:
                microbatches = next(iter_ds).chunk(num_microbatches)
                out = torch.cat([mb.to(device) for mb in microbatches])
                out = net.embed(out)
                send_req = dist.isend(out.chunk(num_microbatches)[microbatch].to("cpu"), dst=pipeline_id * 3 + 1, group=pipeline_group)
                send_req.wait()
            except Exception as e:
                print(f"Pipeline {pipeline_id} Rank {rank} Exception: {e}")
        elif local_rank == 1:
            try:
                expected_shape = (microbatch_size, seq_l, dmodel)
                inp_batch = torch.empty(expected_shape)
                recv_req = dist.irecv(inp_batch, src=pipeline_id * 3 + 0, group=pipeline_group)
                recv_req.wait()
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()
                out = net(inp_batch)
                send_req = dist.isend(out.to("cpu"), dst=pipeline_id * 3 + 2, group=pipeline_group)
                send_req.wait()
            except Exception as e:
                print(f"Pipeline {pipeline_id} Rank {rank} Exception: {e}")
        elif local_rank == 2:
            try:
                target_microbatches = next(iter_ds).chunk(num_microbatches)
                target = target_microbatches[microbatch].to(device)
                inp_batch = torch.empty((microbatch_size, seq_l, dmodel))
                recv_req = dist.irecv(inp_batch, src=pipeline_id * 3 + 1, group=pipeline_group)
                recv_req.wait()
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()
                logits = net(inp_batch)
                loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
                loss.backward(retain_graph=True)
                print(loss.item())
            except Exception as e:
                print(f"Pipeline {pipeline_id} Rank {rank} Exception: {e}")

    # Backward pass
    for microbatch in range(num_microbatches -1, -1, -1):
        print(f"Backward {rank} microbatch {microbatch}")
        if local_rank == 2:
            try: 
                send = dist.isend(inp_batch.grad.to("cpu"), dst=rank-1, group=pipeline_group)
                send.wait()
            except Exception as e:
                print(f"Process {rank} exception {e}")
        elif local_rank == 1:
            try:
                grad = torch.empty(microbatch_size, seq_l, dmodel)
                recv_req = dist.irecv(tensor=grad, src=rank + 1, group=pipeline_group)
                recv_req.wait()
                
                # Accumulate gradients over all microbatches
                inp_batch.grad = torch.zeros_like(inp_batch)

                for mb_out, mb_grad in zip(out.chunk(num_microbatches), grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
                    inp_batch.grad += inp_batch.grad  # Accumulate the gradients correctly

                send_req = dist.isend(inp_batch.grad.to("cpu"), dst=rank - 1, group=pipeline_group) 
                send_req.wait()
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}") 
        elif local_rank == 0:
            try:
                grad = torch.empty(microbatch_size, seq_l, dmodel)
                recv = dist.irecv(tensor=grad, src=rank+1, group=pipeline_group)
                recv.wait()
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}")
    # Synchronize gradients across pipelines
    #for param in net.parameters():
    #    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    optim.step()
    torch.cuda.empty_cache()
    print(f"Pipeline {pipeline_id} Rank {rank} finished iteration {itr}")