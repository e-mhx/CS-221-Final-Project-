word_vec_dict = {}

with open("./data/glove.42B.300d/glove.42B.300d.txt", 'r') as file:

	for lineNum in range(10000):
		line = file.readline()
		lineList = [i for i in line.split()]

		key = lineList[0]
		val = [float(i) for i in lineList[1:]]
		word_vec_dict[key] = val

print word_vec_dict["hello"]

