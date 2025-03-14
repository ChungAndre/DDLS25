from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv

rank = int(argv[1])
os.environ["MASTER_ADDR"] = "localhost"
world_size = 6
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)

num_pipeline = 2
# Create groups 
pipeline_0 = dist.new_group(ranks=[0, 1, 2])
pipeline_1 = dist.new_group(ranks=[3, 4, 5])

layer_groups = [dist.new_group(ranks=[i, i+3]) for i in range(3)]

# Assign pipeline, local rank and group
pipeline_id = rank // 3
local_rank = rank % 3
pipeline_group = pipeline_0 if rank in [0, 1, 2] else pipeline_1
print(f"Process {rank} initialized in pipeline {pipeline_id}, local rank: {local_rank}")

torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 6
device = "cpu"
num_microbatches = 6
microbatch_size = batch_size // num_microbatches

# make the tokenizer

# make the model
if local_rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)
elif local_rank == 1:
    net = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
elif local_rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l) # no skip
    iter_ds = iter(ds)

optim = Adam(net.parameters(),lr=8e-4)

activations = []
gradients = []

for itr in range(5_000):
    print(f"Iteration {itr} started")
    optim.zero_grad()
    # FORWARD PASS:

    for microbatch in range(num_microbatches):
        print(f"Microbatch {microbatch} started")
        if local_rank == 0:
            batch = next(iter_ds)
            microbatches = batch.chunk(num_microbatches)
            out = microbatches[microbatch]
            out = out.to(device)
            out = net.embed(out)
            activations.append(out)

            send = dist.isend(out.to("cpu"), dst=rank+1, group=pipeline_group)
            send.wait()
            print(f"Rank {rank} sent microbatch {microbatch}")

        elif local_rank == 1:
            inp_batch = torch.empty((microbatch_size,seq_l,dmodel))
            recv = dist.irecv(inp_batch,src=rank-1, group=pipeline_group)
            recv.wait()
            print(f"Rank {rank} received microbatch {microbatch}")
            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            out = net(inp_batch)
            activations.append(out)
            send1 = dist.isend(out.to("cpu"),dst=rank+1, group=pipeline_group)
            send1.wait()
            print(f"Rank {rank} sent microbatch {microbatch}")

        elif local_rank == 2:
            target = next(iter_ds)
            target = target.chunk(num_microbatches)[microbatch].to(device)
            inp_batch = torch.empty((microbatch_size,seq_l,dmodel))
            recv = dist.irecv(inp_batch,src=rank-1, group=pipeline_group)
            recv.wait()
            print(f"Rank {rank} received microbatch {microbatch}")
            with torch.no_grad():
                inp_batch = inp_batch.to(device)
                inp_batch.requires_grad_()
                inp_batch.retain_grad()

            logits = net(inp_batch)
            loss = causalLLMLoss(logits,target,tokenizer.vocab_size)
            print(f"Loss for microbatch {microbatch}: {loss.item()}")
            loss.backward()
            gradients.append(inp_batch.grad.clone())  # Store gradient

    # BACKWARD PASS:
    for microbatch in range(num_microbatches -1, -1, -1):
        print(f"Backward microbatch {microbatch} started")
        if local_rank == 2:
            send2 = dist.isend(gradients[microbatch].to("cpu"),dst=rank-1, group=pipeline_group)
            send2.wait()
            print(f"Rank {rank} sent gradient for microbatch {microbatch}")
        elif local_rank == 1:
            inp_grad = torch.empty((microbatch_size,seq_l,dmodel))
            recv = dist.irecv(inp_grad,src=rank + 1, group=pipeline_group)
            recv.wait()
            print(f"Rank {rank} received gradient for microbatch {microbatch}")
            activations[microbatch].backward(inp_grad.to(device), retain_graph=True)
            gradients.append(inp_batch.grad.clone())  # Store gradient
            send1 = dist.isend(inp_batch.grad.to("cpu"),dst=rank - 1, group=pipeline_group)
            send1.wait()
            print(f"Rank {rank} sent gradient to rank 0 for microbatch {microbatch}")
        elif local_rank == 0:
            inp_grad = torch.empty((microbatch_size,seq_l,dmodel))
            recv = dist.irecv(inp_grad,src=rank+1, group=pipeline_group)
            recv.wait()
            print(f"Rank {rank} received gradient for microbatch {microbatch}")
            activations[microbatch].backward(inp_grad.to(device), retain_graph=True)
        
    # Perform all_reduce to aggregate gradients across all processes
    dist.barrier()  # Synchronize all processes before performing all_reduce

    tmp = []  # Temporary list to store flattened gradients
    for param in net.parameters():  # Iterate over all parameters of the model
        if param.grad is None:  # If the parameter has no gradient
            tmp.append(torch.zeros_like(param, device="cpu").view(-1))  # Append a zero tensor of the same shape
            continue
        tmp.append(param.grad.view(-1))  # Flatten the gradient and append to the list
        param.grad = None  # Clear the gradient to avoid accumulation

    prev_grad = torch.cat(tmp).to("cpu")  # Concatenate all flattened gradients into a single tensor and move to CPU
    dist.all_reduce(prev_grad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)  # Perform all_reduce to sum gradients across all processes

    # Split the concatenated tensor back into individual gradients and assign them to the parameters
    tmp = torch.split(prev_grad, [param.numel() for param in net.parameters()])
    for i, param in enumerate(net.parameters()):
        param.grad = tmp[i].view(param.size()).to(device) / world_size  # Reshape and assign the averaged gradient to the parameter

    optim.step()
    torch.cuda.empty_cache()
    activations.clear()
    gradients.clear()
    print(f"Iteration {itr} completed")