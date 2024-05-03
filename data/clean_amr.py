def filter_text(input_filename, output_filename):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            # Check if the line is not part of the annotations
            if line.startswith("#"):
                outfile.write(line)

# Example usage
input_file = 'qald9_test.txt'  # Name of your input file
output_file = 'cleaned_qald9_test.txt'  # Name of your output file
filter_text(input_file, output_file)
