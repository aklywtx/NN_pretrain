import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from tqdm import tqdm


np.random.seed(33)
area_size = 17  # 17˚ × 17˚ area
dot_diameter = 0.26  # Nominal diameter in degrees
s = (dot_diameter / area_size) * 8 * 500  # Approximation of dot size in 's' for scatter

def generate_non_overlapping_points(N, min_distance, max_attempts=1000):
    points = []
    attempts = 0
    
    while len(points) < N and attempts < max_attempts:
        # Generate a new candidate point
        x = np.random.uniform(-8.5, 8.5)
        y = np.random.uniform(-8.5, 8.5)
        
        # Check distance from all existing points
        if not points:  # First point
            points.append((x, y))
            continue
            
        distances = np.sqrt(np.sum(((np.array(points) - [x, y]) ** 2), axis=1))
        if np.all(distances > min_distance):
            points.append((x, y))
        
        attempts += 1
    
    return np.array(points)

# Main loop for varying percentages and dot counts
print("Generating dots.....")
for m in tqdm(range(0, 101), desc="Percentage black"):
    for N in tqdm([100], desc=f"Num dots (m={m})", leave=False):
        for i in range(100):
            # Generate non-overlapping points
            min_distance = dot_diameter * 1.2  # 20% larger than dot diameter to ensure small gap
            points = generate_non_overlapping_points(N, min_distance)
            x, y = points.T  

            # Create a boolean mask for black dots
            num_black_dots = int(N * m / 100)  # Calculate the number of black dots
            # print(num_black_dots)
            black_indices = np.random.choice(N, num_black_dots, replace=False)  # Unique indices for black dots
            black_mask = np.zeros(N, dtype=bool)  # Initialize mask
            black_mask[black_indices] = True  # Set selected indices to True

            plt.figure(figsize=(8, 8), facecolor='#808080')
            plt.scatter(x[~black_mask], y[~black_mask], color='white', s=120, edgecolors='none')
            plt.scatter(x[black_mask], y[black_mask], color='black', s=120, edgecolors='none')

            # Remove axes, borders, labels, and title
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])

            # Set the plot background color to grey
            plt.gca().set_facecolor('#808080')

            # Save the plot to a file
            plt.savefig(f"./test_data/displays_dpi32_ViTtest_only100dots/image_{num_black_dots}_{N}_{i}.png", 
                    dpi=32,
                    bbox_inches='tight',
                    pad_inches=0) 

            plt.close()

# # 10 dots section
# for num_black_dots in tqdm(range(0, 11), desc="10 dots section"):
#     for i in range(100):
#         N = 10
#         # Generate non-overlapping points
#         min_distance = dot_diameter * 1.2  # 20% larger than dot diameter to ensure small gap
#         points = generate_non_overlapping_points(N, min_distance)
#         x, y = points.T  

#         # Create a boolean mask for black dots
#         # num_black_dots = int(N * m / 100)  # Calculate the number of black dots
#         # print(num_black_dots)
#         black_indices = np.random.choice(N, num_black_dots, replace=False)  # Unique indices for black dots
#         black_mask = np.zeros(N, dtype=bool)  # Initialize mask
#         black_mask[black_indices] = True  # Set selected indices to True

#         plt.figure(figsize=(8, 8), facecolor='#808080')
#         plt.scatter(x[~black_mask], y[~black_mask], color='white', s=120, edgecolors='none')
#         plt.scatter(x[black_mask], y[black_mask], color='black', s=120, edgecolors='none')

#         # Remove axes, borders, labels, and title
#         plt.axis('off')
#         plt.gca().set_position([0, 0, 1, 1])

#         # Set the plot background color to grey
#         plt.gca().set_facecolor('#808080')

#         # Save the plot to a file
#         plt.savefig(f"./training_data/displays_dpi32_ViT_only100dots/image_{num_black_dots}_{N}_{i}.png", 
#                 dpi=32,
#                 bbox_inches='tight',
#                 pad_inches=0,
#                 cmap='gray') 

#         plt.close()

# # 50 dots section
# for num_black_dots in tqdm(range(0, 51), desc="50 dots section"):
#     for i in range(100):
#         N = 50
#         # Generate non-overlapping points
#         min_distance = dot_diameter * 1.2  # 20% larger than dot diameter to ensure small gap
#         points = generate_non_overlapping_points(N, min_distance)
#         x, y = points.T  

#         # Create a boolean mask for black dots
#         # num_black_dots = int(N * m / 100)  # Calculate the number of black dots
#         # print(num_black_dots)
#         black_indices = np.random.choice(N, num_black_dots, replace=False)  # Unique indices for black dots
#         black_mask = np.zeros(N, dtype=bool)  # Initialize mask
#         black_mask[black_indices] = True  # Set selected indices to True

#         plt.figure(figsize=(8, 8), facecolor='#808080')
#         plt.scatter(x[~black_mask], y[~black_mask], color='white', s=120, edgecolors='none')
#         plt.scatter(x[black_mask], y[black_mask], color='black', s=120, edgecolors='none')

#         # Remove axes, borders, labels, and title
#         plt.axis('off')
#         plt.gca().set_position([0, 0, 1, 1])

#         # Set the plot background color to grey
#         plt.gca().set_facecolor('#808080')

#         # Save the plot to a file
#         plt.savefig(f"./training_data/displays_dpi32_ViT_only100dots/image_{num_black_dots}_{N}_{i}.png", 
#                 dpi=32,
#                 bbox_inches='tight',
#                 pad_inches=0,
#                 cmap='gray') 

#         plt.close()