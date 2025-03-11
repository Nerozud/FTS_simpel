# import pycuda.driver as cuda
# import pycuda.autoinit

# # Get the device count and the device name
# device_count = cuda.Device.count()
# device_name = cuda.Device(0).name()

# # Print the device information
# print(f"There are {device_count} CUDA devices.")
# print(f"Device 0: {device_name}")




import torch
print("Torch version: ", torch.version.cuda)
print("Cuda available?", torch.cuda.is_available())
print("Cuda device count: ", torch.cuda.device_count())
