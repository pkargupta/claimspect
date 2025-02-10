import argparse
import json

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some dataset.')

    # Add the arguments
    parser.add_argument('dataset', type=str, choices=['vaccine', 'dtra'], help='The name of the dataset', default='dtra')

    # Parse the arguments
    args = parser.parse_args()

    # Use the dataset name to do something
    dataset_name = args.dataset
    print(f'Processing dataset: {dataset_name}')

    # Define the mapping of dataset names to file paths
    dataset_files = {
        'vaccine': 'data/vaccine/segments.json',
        'dtra': 'data/dtra/segments.json'
    }

    # load the json file
    with open(dataset_files[dataset_name], 'r') as file:
        data = json.load(file)

    deduplicated_data = {}
    report = {}

    for id, segments in data.items():
        unique_segments = {}
        for segment in segments:
            seg_text = segment['segment']
            paper_id = segment['paper_id']
            if seg_text not in unique_segments:
                unique_segments[seg_text] = set()
            unique_segments[seg_text].add(paper_id)

        deduplicated_segments = []
        unique_count = 0
        different_paper_id_count = 0

        for seg_text, paper_ids in unique_segments.items():
            unique_count += 1
            if len(paper_ids) > 1:
                different_paper_id_count += 1
            deduplicated_segments.append({
                'segment': seg_text,
                'paper_id': list(paper_ids)[0]  # Keep only one paper_id
            })

        deduplicated_data[id] = deduplicated_segments
        report[id] = {
            'unique_count': unique_count,
            'different_paper_id_count': different_paper_id_count
        }

    # Save the deduplicated data to a new JSON file
    with open('data/deduplicated_segments.json', 'w') as file:
        json.dump(deduplicated_data, file, indent=4)

    # Print the report
    for id, stats in report.items():
        print(f'ID: {id}, Unique segments: {stats["unique_count"]}, Different paper IDs: {stats["different_paper_id_count"]}')

if __name__ == '__main__':
    main()