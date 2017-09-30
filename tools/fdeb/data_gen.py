import argparse
import random


def main():
    n, e = parse_args()
    generate_graph(n, e)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Graph for the FDEB benchmark')

    parser.add_argument('num_nodes', metavar='N', type=int,
                        help='The number of nodes')
    parser.add_argument('num_edges', metavar='E', type=int,
                        help='The number of edges')

    args = parser.parse_args()
    return args.num_nodes, args.num_edges


def generate_graph(n, e):
    random.seed(0)

    #  f_node = open(str(n) + 'x' + str(e) + '_node.data', 'w')
    #  for i in range(0, n):
    #      f_node.write(str(random.random()) + ',' + str(random.random()) + '\n')
    
    scale = 10
    nodes = []
    for i in range(0, n):
        nodes.append((random.random() * scale, random.random() * scale));

    f_edges = open(str(n) + 'x' + str(e) + '.csv', 'w')
    for i in range(0, e):
        src = random.randint(0, n - 1)
        dst = random.randint(0, n - 1)
        f_edges.write(str(nodes[src][0]) + ',' +
                      str(nodes[src][1]) + ',' + 
                      str(nodes[dst][0]) + ',' + 
                      str(nodes[dst][1]) + ',' + 
                      '\n')


if __name__ == '__main__':
    main()
