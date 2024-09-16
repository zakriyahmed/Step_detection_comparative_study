import torch
import numpy as np

def remove_patches(predictions):
    predicted_1_start_end = np.where(np.diff(np.concatenate(([0], predictions == 1, [0]))))[0]
    predicted_middle = torch.tensor((predicted_1_start_end[1::2] + predicted_1_start_end[:-1:2])/2,dtype=torch.int64)
    out = torch.zeros_like(predictions)
    out[[predicted_middle]]=1
    return out#,predicted_middle

def remove_small_brusts(tensor,min_length):

    # Find indices where the tensor values change
    change_points = (tensor[1:] != tensor[:-1]).nonzero(as_tuple=False).flatten() + 1
    change_points = torch.cat((torch.tensor([0]), change_points, torch.tensor([len(tensor)])))
    
    result = tensor.clone()
    
    for start, end in zip(change_points[:-1], change_points[1:]):
        segment = tensor[start:end]
        if segment[0] == 1 and len(segment) < min_length:
            result[start:end] = 0  # Replace short sequences of 1s with 0s
    
    return result


#a = torch.tensor([0,0,0,0,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
#print(remove_patches(remove_small_brusts(a,3)))

 