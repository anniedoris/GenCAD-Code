import matplotlib.pyplot as plt

graph_ques = str(input("Would you like to generate a graph of the truncation level vs number of successfully generated files? (y/n): "))
if graph_ques == 'y':
    generate_graphs = True
else:
    generate_graphs = False

if generate_graphs:
        data = [line.strip().split(",") for line in open(f"{prefix}/trunc_logs.txt")]
        trunc = [float(d[0]) for d in data]
        gen_file = [float(d[1]) for d in data]
        plt.plot(trunc, gen_file, 'ro')
        plt.xlabel('Truncation Level')
        plt.ylabel('Number of Successfully Generated Files')
        plt.show()
