original_txt_path = "data/vaccine/text.txt"

def split_text_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines, start=1):
        output_filename = f"data/vaccine/papersheet_txt/{i}.txt"
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write(line)

if __name__ == "__main__":
    split_text_file(original_txt_path)
