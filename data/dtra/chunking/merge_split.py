import json
import json

input_dir = "data/dtra/chunking/shards"
output_path = "data/dtra/segments.json"

def main():
    
    # load all the json file from the input directory
    all_data = []
    for i in range(0, 16):
        with open(f"{input_dir}/corpus_segments_{i}.json") as f:
            data = json.load(f)
            all_data.append(data)

    # combine the dict
    claims = {}
    for data in all_data:
        for key, value in data.items():
            if key not in claims:
                claims[key] = []
            claims[key].extend(value)
    
    # save the combined dict to the output path
    with open(output_path, "w") as f:
        json.dump(claims, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()