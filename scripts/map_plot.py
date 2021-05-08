import matplotlib.pyplot as plt
import csv

def plot(x, y):
    bbox = (-74.0566, -73.7062, 40.5362, 40.9005)
    nyc = plt.imread('./map.png')
    fig, ax = plt.subplots()
    ax.scatter(x, y, zorder=1, alpha=0.2, c='b', s=10)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.imshow(nyc, zorder=0, extent=bbox, aspect='equal')
    plt.show()
    
def main():
    infile = csv.reader(open('../pts.csv', 'r'))
    x = []
    y = []
    for row in infile:
        x.append(float(row[0]))
        y.append(float(row[1]))
    plot(x, y)

if __name__ == '__main__':
    main()
