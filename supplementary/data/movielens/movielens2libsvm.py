import os

input_files = ["test.csv", "train.csv", "valid.csv"]
output_files = ["test.libsvm", "train.libsvm", "valid.libsvm"]

def WriteOneLine(tokens, output):
	label = int(float(tokens[0]))
	output.write(str(label))
	for i in range(1,len(tokens)):
		feature_value = float(tokens[i])
		output.write(' ' + str(i-1) + ':' + str(feature_value))
	output.write('\n')

for csv_file, libsvm_file in zip(input_files, output_files):
    file_buffer_in = open(csv_file, "r")
    file_buffer_out = open(libsvm_file, "w")
	
    line = file_buffer_in.readline()
    first = 0
    while line:
        if first == 0:
            line = file_buffer_in.readline()
            first = 1
            continue
        tokens = line.split(',')
        WriteOneLine(tokens, file_buffer_out)
        line = file_buffer_in.readline()

    file_buffer_in.close()
    file_buffer_out.close()