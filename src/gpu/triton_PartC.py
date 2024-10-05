import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import numpy as np
import cv2

# Triton kernel for bilateral filtering
@triton.jit
def bilateral_filter_kernel(img_pad_ptr,  # *Pointer* to padded input image
                            output_ptr,  # *Pointer* to output image
                            width, height,
                            pad_size, sigma_space, sigma_density,
                            stride_h_pad, stride_w_pad,
                            stride_h_out, stride_w_out):
    pid_h = tl.program_id(axis=0)  # Block index in height direction
    pid_w = tl.program_id(axis=1)  # Block index in width direction

    offset = (pid_h + pad_size) * stride_h_pad + (pid_w + pad_size) * stride_w_pad
    center_value = tl.load(img_pad_ptr + offset)

    result = 0.
    norm_factor = 0.

    for sub_h in range(-pad_size, pad_size+1):
        for sub_w in range(-pad_size, pad_size+1):
            pixel_value = tl.load(img_pad_ptr + offset + sub_h * stride_h_pad + sub_w * stride_w_pad)
            
            spatial_dist = (sub_h * sub_h + sub_w * sub_w) / (2 * sigma_space * sigma_space)
            intensity_dist = (center_value - pixel_value) ** 2 / (2 * sigma_density * sigma_density)
            
            weight = tl.exp(-spatial_dist - intensity_dist)
            result += pixel_value * weight
            norm_factor += weight

    result /= norm_factor
    output_offset = pid_h * stride_h_out + pid_w * stride_w_out
    tl.store(output_ptr + output_offset, result)

# Function to apply bilateral filter using Triton
def bilateral_filter(img_pad, k_size, sigma_space, sigma_density):
    assert img_pad.is_contiguous(), "Input image must be contiguous"
    
    H, W = img_pad.shape
    pad = (k_size - 1) // 2
    H_orig, W_orig = H - 2 * pad, W - 2 * pad
    
    output = torch.empty((H_orig, W_orig), device=img_pad.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(H_orig, 1), triton.cdiv(W_orig, 1))
    
    bilateral_filter_kernel[grid](
        img_pad, output, W, H, pad, sigma_space, sigma_density,
        img_pad.stride(0), img_pad.stride(1), output.stride(0), output.stride(1)
    )
    return output

def main(input_image_path, output_image_path):
    # Parameters for bilateral filter
    sigma_space = 1.7  # Spatial standard deviation
    sigma_density = 50.0  # Intensity (density) standard deviation
    ksize = 7  # Kernel size

    # Read and preprocess the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    pad = (ksize - 1) // 2
    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    pad_img = torch.tensor(pad_img, device="cuda", dtype=torch.float32)

    # Apply the bilateral filter
    output_triton = bilateral_filter(pad_img, ksize, sigma_space, sigma_density)

    # Save the output image
    cv2.imwrite(output_image_path, output_triton.cpu().numpy())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg")
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])