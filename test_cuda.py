import pycuda.driver as cuda
import pycuda.autoinit

# Get the device count and the device name
device_count = cuda.Device.count()
device_name = cuda.Device(0).name()

# Print the device information
print(f"There are {device_count} CUDA devices.")
print(f"Device 0: {device_name}")




# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
