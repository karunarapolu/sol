import csv
import glob

def csv_to_text(file_path):
    lines = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            line = f"{file_path} - Row {i}: " + ", ".join([f"{key}={value}" for key, value in row.items()])
            lines.append(line)
    return "\n".join(lines)

# Process all CSV files in a folder
all_texts = ["gsoc_2019_projects","gsoc_2020_projects","gsoc_2021_projects","gsoc_2022_projects","gsoc_2023_projects","gsoc_2024_projects","gsoc_2025_projects"]
for file in glob.glob("*.csv"):  # adjust path if needed
    all_texts.append(csv_to_text(file))

# Merge into one big text block
final_text = "\n".join(all_texts)

# Save to TXT
with open("merged_output.txt", "w", encoding="utf-8") as out:
    out.write(final_text)
