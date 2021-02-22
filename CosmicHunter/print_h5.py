#use : python print_h5.py [fileName]
import pandas
import sys

print(pandas.read_hdf(sys.argv[1]))
