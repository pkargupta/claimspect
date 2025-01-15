import json
import matplotlib.pyplot as plt

json_path = "data/dtra/get_literature/claim2paper_meta_info.json"
save_path = "data/dtra/get_literature/claim2paper_meta_info_num_distribution.png"

with open(json_path, 'r') as f:
    claim2paper = json.load(f)

nums = []
for claim, papers in claim2paper.items():
    nums.append(len(papers))

int_list = [int(num) for num in nums]
bin_num = 50


plt.hist(int_list, bins=bin_num, edgecolor='black', alpha=0.7)


plt.title('Distribution of the Number of Papers found for Each Claim')
plt.xlabel('Number of Papers')
plt.ylabel('Number of Claims')

plt.savefig(save_path)
