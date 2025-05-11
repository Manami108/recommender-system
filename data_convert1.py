import ijson
import csv

# Define file paths
json_file_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/dblp.v12.json"
csv_file_path = "/media/e-soc-student/DISK2/GR/GR2_Recommendation/datasets/dblp.v12.csv"

# Open files for streaming and writing
with open(json_file_path, 'r') as f_json, open(csv_file_path, 'w', newline='') as f_csv:
    writer = None  # Initialize the CSV writer
    fieldnames = set()  # Track all unique field names
    count = 0  # Record counter
    first_three_papers = []
    
    # Stream JSON objects from the array
    for record in ijson.items(f_json, "item"):
        # Update fieldnames dynamically
        fieldnames.update(record.keys())

        # Initialize the writer with updated fieldnames if not already done
        if writer is None:
            writer = csv.DictWriter(f_csv, fieldnames=list(fieldnames))
            writer.writeheader()

        # Reinitialize writer if new fields are detected
        if set(writer.fieldnames) != fieldnames:
            f_csv.seek(0)
            writer = csv.DictWriter(f_csv, fieldnames=list(fieldnames))
            f_csv.truncate()
            writer.writeheader()

        # Write the record, filling missing fields with empty values
        writer.writerow({key: record.get(key, "") for key in writer.fieldnames})
        count += 1

        # Store first 3 records
        if count <= 3:
            first_three_papers.append(record)

        # Print progress every 1000 records
        if count % 1000 == 0:
            print(f"Processed {count} records...")

print(f"Converted {count} records to {csv_file_path}")

# Print the total number of records
print(f"Total number of records: {count}")

# Print the first 3 paper details
print("\nFirst 3 papers information:")
for i, paper in enumerate(first_three_papers, start=1):
    print(f"\nPaper {i}:")
    for key, value in paper.items():
        print(f"{key}: {value}")
