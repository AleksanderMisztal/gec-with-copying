import argparse

def main(args):
	m2 = open(args.m2_file).read().strip().split("\n\n")
	out = open(args.out, "w")
	
	for sent in m2:
		sent = sent.split('\n')[0]
		out.write(sent[2:] + '\n')

if __name__ == "__main__":
	# Define and parse program input
	parser = argparse.ArgumentParser()
	parser.add_argument("m2_file", help="The path to an input m2 file.")
	parser.add_argument("-out", help="A path to where we save the output original text file.", required=True)
	args = parser.parse_args()
	main(args)