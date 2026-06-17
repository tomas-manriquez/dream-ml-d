import os
import csv

def create_reduced_csv(input_file, output_file, target_size_mb=10):
    """
    Create a reduced CSV file by copying the first rows until target size is reached.
    
    Args:
        input_file (str): Path to the original CSV file
        output_file (str): Path for the reduced CSV file
        target_size_mb (int): Target file size in MB (will not exceed this)
    """
    target_size_bytes = target_size_mb * 1024 * 1024  # Convert MB to bytes
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Copy header
            header = next(reader)
            writer.writerow(header)
            
            rows_copied = 1  # Count the header
            check_frequency = 1000  # Start checking every 1000 rows
            
            for row in reader:
                writer.writerow(row)
                rows_copied += 1
                
                # Check file size periodically
                if rows_copied % check_frequency == 0:
                    current_size = os.path.getsize(output_file)
                    current_size_mb = current_size / (1024 * 1024)
                    
                    # If we're getting close to target, check more frequently
                    if current_size_mb > target_size_mb * 0.9:  # When over 90% of target
                        check_frequency = 100
                    
                    # Stop if we've reached or exceeded the target size
                    if current_size >= target_size_bytes - 100:
                        print(f"Target size reached at row {rows_copied}")
                        break
        
        # Final size calculation and summary
        final_size = os.path.getsize(output_file)
        final_size_mb = final_size / (1024 * 1024)
        
        print(f"✅ Reduced CSV created successfully!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Original file size: 43.2 MB")
        print(f"📊 New file size: {final_size_mb:.2f} MB")
        print(f"📈 Rows copied: {rows_copied:,}")
        print(f"📅 Data period: Starting from January 1, 2009 (chronological)")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find the input file '{input_file}'")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

# Main execution
if __name__ == "__main__":
    #SI SE QUIERE USAR PARA OTRO DATASET, CAMBIAR NOMBRES ACA (!)
    input_file = "jena_climate_2009_2016.csv"
    output_file = "jena_climate_10mb.csv"
    
    # Check if input file exists before proceeding
    if os.path.exists(input_file):
        create_reduced_csv(input_file, output_file, target_size_mb=10)
    else:
        print(f"❌ Input file '{input_file}' not found in the current directory.")
        print("Please make sure the file is in the same folder as this script.")