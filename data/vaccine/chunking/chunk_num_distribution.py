import json
from matplotlib import pyplot as plt

# Define the paths for the input JSON file and the output image
segment_path = "data/dtra/segments.json"
save_path = "data/dtra/chunking/seg_num_distribution.png"

# Load the JSON file containing segment data
with open(segment_path, "r") as f:
    segments = json.load(f)

# Create a list to store the lengths of each segment
ints = []
for segment in segments:
    ints.append(len(segments[segment]))

# Convert all values to integers and limit the maximum value to 20000
int_list = [min(int(num), 20000) for num in ints]

# Set the number of bins for the histogram
bin_num = 50

# Plot the histogram with specified bins and range
plt.hist(int_list, bins=bin_num, range=(0, 20000), edgecolor='black', alpha=0.7)

# Add title and labels to the plot
plt.title('Distribution of the Relevant Segments found for Each Claim')
plt.xlabel('Number of Segments (20000+ are capped at 20000 in the graph)')
plt.ylabel('Number of Claims')

# Save the histogram plot as a PNG file
plt.savefig(save_path)
