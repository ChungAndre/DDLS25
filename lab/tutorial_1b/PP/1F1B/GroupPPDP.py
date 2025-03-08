from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import TinyStories 
from simplellm.losses import causalLLMLoss
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
print(f"Process {rank} started and waiting for messages...")
torch.manual_seed(0)
dmodel = 288
num_heads = 6
num_group = 2
n_layers = 6 // (world_size // num_group)
seq_l = 256
batch_size = 6
device = "cpu"
num_microbatches = 3
microbatch_size = (batch_size // num_microbatches) // num_group


# Initialize tokenizer and model based on rank
if rank in [0,3]:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)
elif rank in [1,4]:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads, device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank in [2,5]:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)

optimizer = Adam(net.parameters(), lr=8e-4)

for i in range(100):
    print (f"Process {rank} epoch {i}")
    optimizer.zero_grad()

    # Forward pass
    for microbatch in range(num_microbatches):
        print(f"Forward {rank} microbatch {microbatch}")
        if rank in [0,3]:
            try:
                microbatches = next(iter_ds).chunk(num_microbatches * 2) 
                group_microbatches = microbatches[:num_microbatches] if rank == 0 else microbatches[num_microbatches:] 
                out = torch.cat([mb.to(device) for mb in microbatches[microbatch]])
                out = net.embed(out)
                send = dist.isend(out.chunk(num_microbatches)[microbatch].to("cpu"), dst=rank+1)
                send.wait()
            except Exception as e:
                print(f"Process {rank} exception {e}")
        elif rank in [1,4]:
            try:
                inp_batch = torch.empty(microbatch_size, seq_l, dmodel)
                recv = dist.irecv(tensor=inp_batch, src=rank-1)
                recv.wait()
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad = True
                    inp_batch.retain_grad()
                out = net(inp_batch)
                send = dist.isend(out.to("cpu"), dst=rank+1)
                send.wait()
            except Exception as e:
                print(f"Process {rank} exception {e}")
        elif rank in [2,5]:
            try:
                target_microbatches = next(iter_ds).chunk(num_microbatches * 2)  # Create microbatches
                group_microbatches = target_microbatches[:num_microbatches] if rank == 2 else target_microbatches[num_microbatches:] # Split microbatches for each group
                target = group_microbatches[microbatch].to(device) 
                
                inp_batch = torch.empty(microbatch_size, seq_l, dmodel)
                recv = dist.irecv(tensor=inp_batch, src=rank-1)
                recv.wait()
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad = True
                    inp_batch.retain_grad()
                logits = net(inp_batch)
                loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
                print(f"Process {rank} loss {loss}")
                loss.backward()
            except Exception as e:
                print(f"Process {rank} exception {e}")

    
    # Backward pass
    for microbatch in range(num_microbatches -1, -1, -1):
        print(f"Backward {rank} microbatch {microbatch}")
        if rank in [2,5]:
            try: 
                send = dist.isend(inp_batch.grad.to("cpu"), dst=rank-1)
                send.wait()
            except Exception as e:
                print(f"Process {rank} exception {e}")
        elif rank in [1,4]:
            try:
                grad = torch.empty(microbatch_size, seq_l, dmodel)
                recv = dist.irecv(tensor=grad, src=rank+1)
                recv.wait()
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
                    send = dist.isend(inp_batch.grad.to("cpu"), dst=rank - 1) 
                    send.wait() 
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}") 
        elif rank in [0, 3]:
            try:
                grad = torch.empty(microbatch_size, seq_l, dmodel)
                recv = dist.irecv(tensor=grad, src=rank+1)
                recv.wait()
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}")

    optimizer.step()
    torch.cuda.empty_cache()
    print(f"Process {rank} epoch {i} done")
    

            

