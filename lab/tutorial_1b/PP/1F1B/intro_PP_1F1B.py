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
world_size = 3
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=rank, world_size=world_size)
print(f"Process {rank} started and waiting for messages...")
torch.manual_seed(0)
dmodel = 288
num_heads = 6
n_layers = 6 // world_size
seq_l = 256
batch_size = 3
device = "cpu"
num_microbatches = 3
microbatch_size = batch_size // num_microbatches

# Initialize tokenizer and model based on rank
if rank == 0:
    tokenizer = SPTokenizer()
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)
elif rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l)
elif rank == 2:
    tokenizer = SPTokenizer()
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l)
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l)
    iter_ds = iter(ds)

optim = Adam(net.parameters(), lr=8e-4)


for itr in range(5_000):
    print(f"Iteration {itr} started")
    optim.zero_grad()

    # Forward pass
    # Separate batches into microbatches
    for microbatch in range(num_microbatches):
        print(f"Microbatch {microbatch} started")
        if rank == 0:
            try:
                # Get the next batch
                batch = next(iter_ds)
                # Chunk the batch into microbatches 
                microbatches = batch.chunk(num_microbatches)
                mb = microbatches[microbatch].to(device)
                # Embed the microbatch
                out = net.embed(mb)
                send = dist.isend(mb.to("cpu"), dst=1)
                #send.wait() wait at the end.
                """
                # Concatenate the microbatches and move them to the device
                 out = torch.cat([mb.to(device) for mb in microbatches])
                # Embed the microbatches
                out = net.embed(out)
                print(f"Rank 0 out shape: {out.chunk(num_microbatches)[microbatch].shape}")
                # Send the current microbatch to the next rank
                send_req = dist.isend(out.chunk(num_microbatches)[microbatch].to("cpu"), dst=1)
                print(f"Rank 0 waiting to send output.")
                send_req.wait()
                """
                print("Rank 0 forward pass done.")
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}")

        elif rank == 1:
            try:
                expected_shape = (microbatch_size, seq_l, dmodel)
                inp_batch = torch.empty(expected_shape)
                print(f"Rank 1 expecting shape: {inp_batch.shape}")
                recv_req = dist.irecv(inp_batch, src=0)
                recv_req.wait()
                print(f"Rank 1 received shape: {inp_batch.shape}")

                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()
                out = net(inp_batch)

                print(f"Rank 1 out shape: {out.shape}")
                print(f"Rank {rank} forward pass done.")
                send_req = dist.isend(out.to("cpu"), dst=2)
                send_req.wait()
                print("Rank 1 forward pass done.")
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}")

        elif rank == 2:
            try:
                target_microbatches = next(iter_ds).chunk(num_microbatches)
                target = target_microbatches[microbatch].to(device)
                inp_batch = torch.empty((microbatch_size, seq_l, dmodel))
                recv_req = dist.irecv(inp_batch, src=1)
                print(f"Rank 2 waiting to receive input batch.")
                recv_req.wait()
                print(f"Rank {rank} received input batch.")
                with torch.no_grad():
                    inp_batch = inp_batch.to(device)
                    inp_batch.requires_grad_()
                    inp_batch.retain_grad()
                logits = net(inp_batch)
                loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
                print(f"Rank {rank} loss: {loss.item()}")
                loss.backward(retain_graph=True)
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}")

    # Backward pass
    for microbatch in range(num_microbatches -1, -1, -1):
        print(f"Backward microbatch {microbatch} started")
        if rank == 2:
            try:
                req_send = dist.isend(inp_batch.grad.to("cpu"), dst=1)
                req_send.wait()
                print("Rank 2 backward pass done.")
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}")
        elif rank == 1:
            try:
                inp_grad = torch.empty((microbatch_size, seq_l, dmodel))
                req_recv= dist.irecv(inp_grad, src=2)
                print(f"Rank 1 waiting to receive gradients.")
                req_recv.wait()
                print(f"Rank {rank} received gradients.")
                # Accumulate gradients over all microbatches
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
                    req_send = dist.isend(inp_batch.grad.to("cpu"), dst=0)
                    req_send.wait()
                print("Rank 1 backward pass done.")
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}")
        elif rank == 0:
            try:
                inp_grad = torch.empty((microbatch_size, seq_l, dmodel))
                req_recv= dist.irecv(inp_grad, src=1)
                print(f"Rank 0 waiting to receive gradients.")
                req_recv.wait()
                print(f"Rank {rank} received gradients.")
                # Accumulate gradients over all microbatches
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):
                    mb_out.backward(mb_grad.to(device), retain_graph=True)
                print("Rank 0 backward pass done.")
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}")

    optim.step()
    torch.cuda.empty_cache()
    print(f"Iteration {itr} completed")