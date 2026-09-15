# imports
import sys
import numpy as np
from ersilia_pack_utils.core import read_smiles, write_out
from predict import ORGANISMS, predict_batch

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = predict_batch(smiles_list)

# check input and output have the same length
assert len(smiles_list) == outputs.shape[0]

header = [f"{organism}_activity" for organism in ORGANISMS]

# write output in a .csv file
write_out(outputs, header, output_file, np.float32)
