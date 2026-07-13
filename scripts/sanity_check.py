import numpy as np

# (comment removed)
file_path = "/media/backup_SSD/ASPED_v3_npy/Session_02152024/IntersectionB/Audio/recorder_DR-05X-01/0001.npy" 

# (comment removed)
data = np.load(file_path)

print(f"File: {file_path}")
print(f"Shape: {data.shape}") # 1?? ??(Audio)?? 2??(Features)?? ??
print(f"Data Type: {data.dtype}")

# (comment removed)
duration_sec = data.shape[0] / 16000
print(f"Total Duration: {duration_sec:.2f} seconds")