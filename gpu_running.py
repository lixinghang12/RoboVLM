import torch

device = torch.device('cuda')
x = torch.randn(3*10**8,3,3).to(device)
y = torch.randn(3*10**8,3,3).to(device)
while True:
    z = x @ y
    del z    
    torch.cuda.empty_cache()