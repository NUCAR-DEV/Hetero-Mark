import pandas as pd
import matplotlib as mtp
mtp.use('agg')
import matplotlib.pyplot as plt

def main():
    data_name = parse_args()
    nodes, edges = load_data(data_name)
    lines = process_data(nodes, edges)
    plot(lines)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Graph for the FDEB benchmark')

    parser.add_argument('data_name', metavar='N', type=int,
                        help='The number of nodes')

    args = parser.parse_args()
    return args.num_nodes, args.num_edges


    pass

def load_data(data_name):
    nodes = pd.read_csv(data_name + '_node.data', header=None) \
            .rename(columns={0:'x', 1:'y'})
    edges = pd.read_csv(data_name + '_edge.data', header=None) \
            .rename(columns={0:'src', 1:'dst'})

    return nodes, edges

def process_data(edges, nodes):
    edges = edges.join(nodes, on='src', rsuffix='_src')
    edges = edges.join(nodes, on='dst', rsuffix='_dst')

    lines = []
    def iterate_row(data):
        lines.append([(data['x'], data['y']), (data['x_dst'], data['y_dst'])])
    edges.apply(iterate_row, axis=1)

    return lines

def plot(lines):
    line_collections = mtp.collections.LineCollection(lines)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(line_collections)
    plt.savefig('out.eps')

if __name__ == '__main__':
    main()
