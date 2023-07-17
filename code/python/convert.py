import sys, getopt
from extract import convert_counts_csv

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('convert.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('convert.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   convert_counts_csv(file = inputfile, outfile = outputfile, 
   freq = 80, epoch = 60, verbose = True, time_column = "HEADER_TIMESTAMP")

if __name__ == "__main__":
   main(sys.argv[1:])
