import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim
import math

def animate_comparison(A, B, case_index, output_file, interval=200):
    """
    Create and save an animation showing A and B side by side for a specific case.

    Parameters:
        A (numpy.ndarray): Array with shape (n_cases, n_t, n_x, n_y).
        B (numpy.ndarray): Array with shape (n_cases, n_t, n_x, n_y).
        case_index (int): Index of the case to animate.
        output_file (str): Path to save the .mp4 file.
        interval (int): Delay between frames in milliseconds.

    Returns:
        None
    """
    n_cases, n_t, n_x, n_y = A.shape
    assert A.shape == B.shape, "A and B must have the same shape."
    assert 0 <= case_index < n_cases, f"case_index must be between 0 and {n_cases - 1}."

    # Extract the data for the selected case
    a_case = A[case_index]
    b_case = B[case_index]

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
    ax_a, ax_b = axes
    ax_a.set_title("Ground truth")
    ax_b.set_title("Prediction")
    ax_a.set_xlabel('x')
    ax_a.set_ylabel('y')
    ax_b.set_xlabel('x')

    # Initialize the images with 90-degree rotation
    im_a = ax_a.imshow(np.rot90(a_case[0], k=1), cmap='inferno', aspect='equal', vmin=0, vmax=1)
    im_b = ax_b.imshow(np.rot90(b_case[0], k=1), cmap='inferno', aspect='equal', vmin=0, vmax=1)

    # Set the axes to be equal and remove ticks
    ax_a.axis("equal")
    ax_a.axis("off")
    ax_b.axis("equal")
    ax_b.axis("off")

    # Update function for animation
    def update(frame):
        im_a.set_data(np.rot90(a_case[frame], k=1))
        im_b.set_data(np.rot90(b_case[frame], k=1))
        return im_a, im_b

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=n_t, interval=interval, blit=True)

    # Save the animation as an MP4 file
    ani.save(output_file, writer='ffmpeg', fps=1000 // interval)

    # Close the figure
    plt.close(fig)
    
from skimage.metrics import structural_similarity as ssim
import math

def find_furthest_fire_location(matrix):
    
    # Find the last index along axis 0 where values exceed 0.5 for each column
    exceed_indices = np.where(matrix > 0.5, np.arange(matrix.shape[0])[:, None], -1)
    
    # Find the maximum index where values exceed 0.5 for each column
    max_indices_per_column = exceed_indices.max(axis=0)
    
    # Find the largest of these indices
    return max_indices_per_column.max()

def calculate_centroid(matrix):
    """
    Calculate the centroid of a 2D matrix.
    
    Parameters:
        matrix (np.ndarray): Input 2D matrix.
    
    Returns:
        tuple: (x_centroid, y_centroid), the x and y coordinates of the centroid.
    """
    total_weight = matrix.sum()
    if total_weight == 0:
        return (None, None)  # Centroid is undefined for a zero-weight matrix
    
    # Create index grids for x and y
    x_indices = np.arange(matrix.shape[0])[:, None]  # Shape (113, 1)
    y_indices = np.arange(matrix.shape[1])[None, :]  # Shape (1, 32)
    
    # Calculate weighted sum of indices for x and y
    x_centroid = (matrix * x_indices).sum() / total_weight
    y_centroid = (matrix * y_indices).sum() / total_weight
    
    return x_centroid, y_centroid

# Assuming A and B have the shape (n_cases, n_t, n_x, n_y)
# a_init and b_init has shape of (n_x, n_y)
# It assumes A is the ground truth for percentage error calculation (denominator selection)
def calculate_metrics(A, B, a_init=None, b_init=None):
    n_t, n_x, n_y = A.shape
    
    # Initialize arrays for metrics
    mse_values = np.zeros(n_t)
    ssim_values = np.zeros(n_t)
    jaccard_values = np.zeros(n_t)
    fire_loc_values = np.zeros(n_t)
    perc_fire_loc_values = np.zeros(n_t)
    mean_fire_values = np.zeros(n_t)
    perc_mean_fire_values = np.zeros(n_t)
    spread_loc_values = np.zeros(n_t)
    perc_spread_loc_values = np.zeros(n_t)
    spread_mean_values = np.zeros(n_t)
    perc_spread_mean_values = np.zeros(n_t)
    
    for t in range(n_t):
        if math.isnan(B[t, 0, 0]):
            mse_values[t] = np.nan
            ssim_values[t] = np.nan
            jaccard_values[t] = np.nan
            continue
        a = A[t, :, :]
        b = B[t, :, :]
        
        if np.isnan(a).any() or np.isnan(b).any():
            mse = np.nan
            ssim_value = np.nan
            jaccard_value = np.nan
            fire_loc_error = np.nan
            mean_fire_line_error = np.nan
        else:
            # MSE
            mse = np.mean((a - b) ** 2)
            
            # SSIM
            ssim_value = ssim(a, b, data_range=b.max() - b.min())
            
            # Jaccard (assume that all values of b > 0.5 indicates the fire position)
            intersection = np.logical_and(a, b > 0.5).sum()
            union = np.logical_or(a, b > 0.5).sum()
            jaccard_value = intersection / union if union > 0 else 1.0
                        
            # Error of the furthest fire line location
            furthest_a = find_furthest_fire_location(a)
            furthest_b = find_furthest_fire_location(b)
            fire_loc_error = np.abs(furthest_a - furthest_b)*8 #multiply by 8 because that's the filter size
            
            perc_fire_loc_error = fire_loc_error/(furthest_a*8-200)
            if np.isinf(perc_fire_loc_error):
                perc_fire_loc_error = np.nan
            
            # Error of the mean fire spread (defined by furthest fire line)
            if t==0:
                prev_furthest_a = find_furthest_fire_location(a_init)
                prev_furthest_b = find_furthest_fire_location(b_init)
            fire_spread_loc_error = np.abs((furthest_a - prev_furthest_a)*8/1 - (furthest_b - prev_furthest_b)*8/1)
            prev_furthest_a = furthest_a
            prev_furthest_b = furthest_b
            
            perc_fire_spread_loc_error = fire_spread_loc_error/((furthest_a - prev_furthest_a)*8/1)
            if np.isinf(perc_fire_spread_loc_error):
                perc_fire_spread_loc_error = np.nan
            
            # Error of the mean fire line location. Note that we need to ensure that a and b are non-negative for centroid calculation
            a_in = a
            b_in = b
            a_in[a_in<0] = 0
            b_in[b_in<0] = 0
            x_a,_ = calculate_centroid(a)
            x_b,_ = calculate_centroid(b)
            mean_fire_line_error = np.abs(x_a - x_b)*8
            
            perc_mean_fire_line_error = mean_fire_line_error/(x_a*8 - 200)
            if np.isinf(perc_mean_fire_line_error):
                perc_mean_fire_line_error = np.nan
            
            # Error of the mean fire spread (defined by mean fire)
            if t==0:
                prev_x_a,_ = calculate_centroid(a_init)
                prev_x_b,_ = calculate_centroid(b_init)
            fire_spread_mean_error = np.abs((x_a-prev_x_a)/1 - (x_b - prev_x_b)/1)*8
            prev_x_a = x_a
            prev_x_b = x_b
            
            perc_fire_spread_mean_error = fire_spread_mean_error/((x_a-prev_x_a)*8/1)
            if np.isinf(perc_fire_spread_mean_error):
                perc_fire_spread_mean_error = np.nan
        
        mse_values[t] = mse
        ssim_values[t] = ssim_value
        jaccard_values[t] = jaccard_value
        fire_loc_values[t] = fire_loc_error
        perc_fire_loc_values[t] = perc_fire_loc_error*100
        spread_loc_values[t] = fire_spread_loc_error
        perc_spread_loc_values[t] = perc_fire_spread_loc_error*100
        mean_fire_values[t] = mean_fire_line_error
        perc_mean_fire_values[t] = perc_mean_fire_line_error*100
        spread_mean_values[t] = fire_spread_mean_error
        perc_spread_mean_values[t] = perc_fire_spread_mean_error*100
            
    return mse_values, ssim_values, jaccard_values, fire_loc_values, perc_fire_loc_values, mean_fire_values, perc_mean_fire_values, spread_loc_values, perc_spread_loc_values, spread_mean_values, perc_spread_mean_values

def find_fire_scar(matrix,initial_fire_line):
    fire_scar = np.zeros(np.shape(matrix))
    
    # find the initial fire scar
    for indy in range(np.shape(matrix)[2]):
        # make a line for the initial fire scar
        for indx in range(np.shape(matrix)[1]):
            if indx*8 >= 200:
                fire_scar[0,indx,indy] = 1
                break
        
        # find the initial fire scar
        for indx in range(np.shape(matrix)[1]):
            if indx*8 < 200:
                continue
            elif initial_fire_line[indx,indy] > 0.5:
                fire_scar[0,indx,indy] = 1
                break
            else:
                fire_scar[0,indx,indy] = 1
    
    # generate the fire scar as the flame propagates in time
    for indt in range(1, np.shape(matrix)[0]):
        if np.isnan(matrix[indt-1,:,:]).any():
            fire_scar[indt,:,:] = np.nan
        else:
            fire_scar[indt,:,:] = fire_scar[indt-1,:,:]
            burning_now = (matrix[indt-1,:,:] > 0.5).astype(int)
            fire_scar[indt,:,:] = np.clip(fire_scar[indt,:,:] + burning_now, 0, 1)
    return fire_scar