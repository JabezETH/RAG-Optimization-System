import sys
import json
sys.path.insert(1, '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/scripts')
import file_loader 
import evaluation


# Load JSON from file
json_path = '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/filepath.json'

with open(json_path, 'r') as json_file:
    file_paths = json.load(json_file)
data_file_path = file_paths['data_file_path']
synthetic_test_data_path = file_paths['synthetic_test_data_path']

# loading data
data = file_loader.load_csv(data_file_path)

# Generate syntetic test data
syntetic_test_data =evaluation.generate_syntetic_testdata(data, file_path= '/home/jabez/rizzbuzz with poetry/RAG-Optimization-System/test_data')
