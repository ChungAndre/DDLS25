from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # Import model stages
from simplellm.tokenizers import SPTokenizer # Import tokenizer
from simplellm.dataloaders import TinyStories # Import dataset
from simplellm.losses import causalLLMLoss # Import loss function
from torch.optim import SGD, Adam # Import optimizers
import torch.nn.functional as F # Import functional API from PyTorch
import torch # Import PyTorch
import torch.distributed as dist # Import PyTorch distributed package
import os # Import OS module
from sys import argv # Import argv from sys module

rank = int(argv[1]) # Get the rank of the current process from command line arguments
os.environ["MASTER_ADDR"] = "localhost" # Set the master address for distributed training
world_size = 3 # Set the number of processes
os.environ["MASTER_PORT"] = "29500" # Set the master port for distributed training
dist.init_process_group("gloo", rank=rank, world_size=world_size) # Initialize the process group for distributed training
print(f"Process {rank} started and waiting for messages...") # Print a message indicating the process has started
torch.manual_seed(0) # Set the random seed for reproducibility
dmodel = 288 # Set the model dimension
num_heads = 6 # Set the number of attention heads
n_layers = 6 // world_size # Set the number of layers per process
seq_l = 256 # Set the sequence length
batch_size = 3 # Set the batch size
device = "cpu" # Set the device to CUDA
num_microbatches = 3 # Set number of microbatches
microbatch_size = batch_size // num_microbatches # Set microbatch size

# Initialize tokenizer and model based on rank
if rank == 0:
    tokenizer = SPTokenizer() # Initialize tokenizer
    net = LLamaFirstStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                          device=device, n_layers=n_layers, ctx_size=seq_l) # Initialize first stage model
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l) # Initialize dataset
    iter_ds = iter(ds) # Create an iterator for the dataset
elif rank == 1:
    net = LLamaStage(dmodel=dmodel, num_heads=num_heads,
                     device=device, n_layers=n_layers, ctx_size=seq_l) # Initialize middle stage model
elif rank == 2:
    tokenizer = SPTokenizer() # Initialize tokenizer
    net = LLamaLastStage(tokenizer.vocab_size, dmodel=dmodel, num_heads=num_heads,
                         device=device, n_layers=n_layers, ctx_size=seq_l) # Initialize last stage model
    ds = TinyStories(tokenizer, batch_size=batch_size, seq_l=seq_l) # Initialize dataset
    iter_ds = iter(ds) # Create an iterator for the dataset

optim = Adam(net.parameters(), lr=8e-4) # Initialize optimizer

for itr in range(5_000): # Training loop
    print(f"Iteration {itr} started") # Debug print
    optim.zero_grad() # Zero the gradients
    for microbatch in range(num_microbatches): # Microbatch loop
        print(f"Microbatch {microbatch} started") # Debug print

        # Forward pass
        if rank == 0:
            try:
                microbatches = next(iter_ds).chunk(num_microbatches) # Get next batch from dataset
                out = torch.cat([mb.to(device) for mb in microbatches])  # Convert each tensor to device and concatenate
                out = net.embed(out) # Embed the batch
                print(f"Rank 0 out shape: {out.chunk(num_microbatches)[microbatch].shape}") # Debug print
                send_req = dist.isend(out.chunk(num_microbatches)[microbatch].to("cpu"), dst=1) # Send output to next stage
                print(f"Rank 0 waiting to send output.") # Debug print
                send_req.wait()  # Ensure the data is sent before proceeding
                print("Rank 0 forward pass done.") # Debug print
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}") # Print an error message if an exception occurs

        elif rank == 1:
            try:
                expected_shape = (microbatch_size, seq_l, dmodel)  # Ensure it matches the microbatch
                inp_batch = torch.empty(expected_shape)
                print(f"Rank 1 expecting shape: {inp_batch.shape}")  # Debug
                recv_req = dist.irecv(inp_batch, src=0)
                recv_req.wait()
                print(f"Rank 1 received shape: {inp_batch.shape}")  # Debug

                with torch.no_grad():
                    inp_batch = inp_batch.to(device) # Move input batch to device
                    inp_batch.requires_grad_() # Enable gradient computation
                    inp_batch.retain_grad() # Retain gradients for backward pass
                out = net(inp_batch) # Forward pass through the model
                print(f"Rank 1 out shape: {out.shape}") # Debug print
                print(f"Rank {rank} forward pass done.") # Debug print
                send_req = dist.isend(out.to("cpu"), dst=2) # Send output to next stage
                send_req.wait()  # Ensure the data is sent before proceeding
                print("Rank 1 forward pass done.") # Debug print
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}") # Print an error message if an exception occurs

        elif rank == 2:
            try:
                target_microbatches = next(iter_ds).chunk(num_microbatches)  # Create microbatches
                target = target_microbatches[microbatch].to(device)  # Select the correct one
                # Convert each tensor to device and concatenate
                inp_batch = torch.empty((microbatch_size, seq_l, dmodel)) # Create empty tensor for input batch
                recv_req = dist.irecv(inp_batch, src=1) # Receive input batch from previous stage
                print(f"Rank 2 waiting to receive input batch.") # Debug print
                recv_req.wait() # Ensure the data is received before proceeding
                print(f"Rank {rank} received input batch.") # Debug print
                with torch.no_grad():
                    inp_batch = inp_batch.to(device) # Move input batch to device
                    inp_batch.requires_grad_() # Enable gradient computation
                    inp_batch.retain_grad() # Retain gradients for backward pass
                logits = net(inp_batch) # Forward pass through the model
                loss = causalLLMLoss(logits, target, tokenizer.vocab_size) # Compute loss
                print(f"Rank {rank} loss: {loss.item()}") # Print the loss
                loss.backward(retain_graph=True) # Backward pass with retain_graph=True
            except Exception as e:
                print(f"Rank {rank} forward pass error: {e}") # Print an error message if an exception occurs

    for microbatch in range(num_microbatches -1, -1, -1): # Microbatch loop
        print(f"Backward microbatch {microbatch} started") # Debug print
        # Backward pass
        if rank == 2:
            try:
                req_send = dist.isend(inp_batch.grad.to("cpu"), dst=1) # Send gradients to previous stage
                req_send.wait() # Ensure the gradients are sent before proceeding
                print("Rank 2 backward pass done.") # Debug print
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}") # Print an error message if an exception occurs
        elif rank == 1:
            try:
                inp_grad = torch.empty((microbatch_size, seq_l, dmodel)) # Create empty tensor for input gradients
                req_recv= dist.irecv(inp_grad, src=2) # Receive gradients from next stage
                print(f"Rank 1 waiting to receive gradients.") # Debug print
                req_recv.wait() # Ensure the gradients are received before proceeding
                print(f"Rank {rank} received gradients.") # Debug print
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True) # Backward pass through the model
                    req_send = dist.isend(inp_batch.grad.to("cpu"), dst=0) # Send gradients to previous stage
                    req_send.wait() # Ensure the data is sent before proceeding
                print("Rank 1 backward pass done.") # Debug print
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}") # Print an error message if an exception occurs
        elif rank == 0:
            try:
                inp_grad = torch.empty((microbatch_size, seq_l, dmodel)) # Create empty tensor for input gradients
                req_recv= dist.irecv(inp_grad, src=1) # Receive gradients from next stage
                print(f"Rank 0 waiting to receive gradients.") # Debug print
                req_recv.wait() # Ensure the gradients are received before proceeding
                print(f"Rank {rank} received gradients.") # Debug print
                for mb_out, mb_grad in zip(out.chunk(num_microbatches), inp_grad.chunk(num_microbatches)):  
                    mb_out.backward(mb_grad.to(device), retain_graph=True) # Backward pass through the model
                print("Rank 0 backward pass done.") # Debug print
            except Exception as e:
                print(f"Rank {rank} backward pass error: {e}") # Print an error message if an exception occurs

    optim.step() # Update model parameters
    torch.cuda.empty_cache() # Clear CUDA cache
    print(f"Iteration {itr} completed") # Debug print