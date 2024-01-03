import os
import torch
from torchvision import datasets, transforms

def process_data():
    # Step 1: Read the raw data
    raw_data_path = 'MyProject/data/raw'
    file_names = os.listdir(raw_data_path)
    
    # Assuming the files are in .pt format
    data = []
    for file_name in file_names:
        file_path = os.path.join(raw_data_path, file_name)
        tensor = torch.load(file_path)
        data.append(tensor)
    
    # Step 2: Concatenate tensors into a single tensor
    processed_data = torch.cat(data, dim=0)  # Assuming tensors have the same shape
    
    # Step 3: Normalization
    mean = processed_data.mean()
    std = processed_data.std()
    normalized_data = (processed_data - mean) / std

    # Step 4: Save processed data
    processed_data_path = 'MyProject/data/processed'
    os.makedirs(processed_data_path, exist_ok=True)
    torch.save(normalized_data, os.path.join(processed_data_path, 'processed_data.pt'))

if __name__ == "__main__":
    process_data()
