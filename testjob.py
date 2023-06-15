import sys
import torch

# TORCH DEVICES
device_str = "cuda" if torch.cuda.is_available() else "cpu"
# device_str = "cpu"
device = torch.device(device_str)
# torch.cuda.set_device(0)
print("Device type:", device_str, device)

print("torch was successfully imported")
print("arguments:")
for i, arg in enumerate(sys.argv):
    print(i, arg)