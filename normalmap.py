import cv2 as cv
import numpy as np
import scipy.io
import glob

def load_mat_images(mat_files_pattern):
    """Load images from .mat files"""
    mat_files = sorted(glob.glob(mat_files_pattern))
    images = []
    
    for mat_file in mat_files:
        print(f"Loading: {mat_file}")
        mat_data = scipy.io.loadmat(mat_file)
        
        # Find the image data (skip MATLAB metadata keys)
        image_data = None
        for key, value in mat_data.items():
            if not key.startswith('__') and isinstance(value, np.ndarray):
                if len(value.shape) >= 2:
                    image_data = value
                    if len(value.shape) == 3:
                        image_data = value[:,:,0]  # Take first channel if RGB
                    break
        
        if image_data is None:
            raise ValueError(f"No image data found in {mat_file}")
        
        # Ensure uint8 format
        if image_data.dtype != np.uint8:
            if image_data.max() <= 1.0:
                image_data = (image_data * 255).astype(np.uint8)
            else:
                image_data = ((image_data - image_data.min()) / 
                            (image_data.max() - image_data.min()) * 255).astype(np.uint8)
        
        images.append(image_data)
        print(f"Image shape: {image_data.shape}")
    
    return images

def load_light_directions(light_file):
    """Load light directions from text file"""
    light_directions = []
    with open(light_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                values = [float(x) for x in line.split()]
                if len(values) >= 3:
                    light_directions.append(values[:3])
    
    light_mat = np.array(light_directions, dtype=np.float32)
    
    # Normalize each light direction
    for i in range(len(light_mat)):
        norm = np.linalg.norm(light_mat[i])
        if norm > 0:
            light_mat[i] = light_mat[i] / norm
    
    return light_mat

def generate_normal_map(images, light_mat, mask=None):
    """Generate normal map using photometric stereo - extracted from your original code"""
    
    print("Running normal map generation...")
    num_images = len(images)
    
    # Apply mask if provided
    if mask is not None:
        for i in range(num_images):
            images[i] = np.multiply(images[i], mask/255)

    # Convert to float32 (from your original code)
    input_arr_conv = []
    for i in range(num_images):
        im_fl = np.float32(images[i])
        im_fl = im_fl / 255
        input_arr_conv.append(im_fl)
    
    h = input_arr_conv[0].shape[0]
    w = input_arr_conv[0].shape[1]
    
    # Initialize arrays (from your original code)
    normalmap = np.zeros((h, w, 3), dtype=np.float32)
    pgrads = np.zeros((h, w), dtype=np.float32)
    qgrads = np.zeros((h, w), dtype=np.float32)
    
    # Compute pseudo-inverse (from your original code)
    lpinv = np.linalg.pinv(light_mat)
    
    # Reshape intensities (from your original code)
    intensities = []
    norm = []
    for imid in range(num_images):
        a = np.array(input_arr_conv[imid]).reshape(-1)
        intensities.append(a)
    intensities = np.array(intensities)
    
    # Core photometric stereo computation (from your original code)
    rho_z = np.einsum('ij,jk->ik', lpinv, intensities)
    rho = rho_z.transpose()
    norm.append(np.sum(np.abs(rho)**2, axis=-1)**(1./2))
    norm_t = np.array(norm).transpose()
    norm_t = np.clip(norm_t, 0, 1)
    norm_t = np.where(norm_t==0, 1, norm_t)
    
    # Compute albedo (from your original code)
    albedo = np.reshape(norm_t, (h, w))
    
    # Normalize rho (from your original code)
    rho = np.divide(rho, norm_t)
    rho[:, 2] = np.where(rho[:, 2] == 0, 1, rho[:, 2])
    rho = np.asarray(rho).transpose()
    
    # Build normal map (from your original code)
    normalmap[:, :, 0] = np.reshape(rho[0], (h, w))
    normalmap[:, :, 1] = np.reshape(rho[1], (h, w))
    normalmap[:, :, 2] = np.reshape(rho[2], (h, w))
    
    # Compute gradients (from your original code)
    pgrads[0:h, 0:w] = normalmap[:, :, 0] / normalmap[:, :, 2]
    qgrads[0:h, 0:w] = normalmap[:, :, 1] / normalmap[:, :, 2]
    
    # Format for display (from your original code)
    normalmap = normalmap.astype(np.float32)
    normalmap = cv.cvtColor(normalmap, cv.COLOR_BGR2RGB)
    output_int = cv.normalize(normalmap, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC3)
    
    if mask is not None:
        output_int = cv.bitwise_and(output_int, output_int, mask=mask)
        normalmap = cv.bitwise_and(normalmap, normalmap, mask=mask)
    
    print("Normal map computation completed")
    return normalmap, albedo, output_int

# Main execution
if __name__ == "__main__":
    # Set folder path
    folder_path = "queen"
    
    # Load your .mat files from queen folder
    mat_pattern = f"{folder_path}/*.mat"
    images = load_mat_images(mat_pattern)
    
    # Load light directions from queen folder
    light_file = f"{folder_path}/light_directions.txt"
    light_mat = load_light_directions(light_file)
    
    # Load mask if available
    mask_path = f"{folder_path}/mask.png"
    mask = None
    try:
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        print(f"Loaded mask from {mask_path}")
    except:
        print("No mask found, processing without mask")
    
    print(f"Loaded {len(images)} images")
    print(f"Light matrix shape: {light_mat.shape}")
    
    # Generate normal map with mask
    normal_map, albedo, normal_display = generate_normal_map(images, light_mat, mask)
    
    # Save results to queen folder
    output_folder = folder_path
    cv.imwrite(f"{output_folder}/normal_map.png", normal_map.astype(np.float32))
    cv.imwrite(f"{output_folder}/normal_map_display.png", normal_display)
    cv.imwrite(f"{output_folder}/albedo.png", albedo.astype(np.float32))
    cv.imwrite(f"{output_folder}/albedo_display.png", (albedo * 255).astype(np.uint8))
    
    # Resize display outputs for visualization
    resize_width = 600
    resize_height = 600
    normal_display_resized = cv.resize(normal_display, (resize_width, resize_height), interpolation=cv.INTER_AREA)
    albedo_resized = cv.resize((albedo * 255).astype(np.uint8), (resize_width, resize_height), interpolation=cv.INTER_AREA)
    
    # Display results
    cv.imshow('Normal Map', normal_display_resized)
    cv.imshow('Albedo', albedo_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    print(f"Results saved to '{output_folder}/' folder:")
    print("- normal_map.png (high precision)")
    print("- normal_map_display.png (visualization)")
    print("- albedo.png (high precision)")
    print("- albedo_display.png (visualization)")