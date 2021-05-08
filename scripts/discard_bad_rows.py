import csv
def main():
    infile = csv.reader(open('../data/yellow_tripdata_2015-01.csv', 'r'))
    outfile = csv.writer(open('../data/CLEANED_yellow_tripdata_2015-01.csv', 'w'))
    hdr = infile.__next__()
    outfile.writerow(hdr)
    skipped = 0
    total = 0
    for row in infile:
        dropoff_lon, dropoff_lat = float(row[9]), float(row[10])
        pickup_lon, pickup_lat = float(row[5]), float(row[6])
        # discard anything which clearly isn't in NYC
        if dropoff_lon > -75 and dropoff_lon < -73 and \
           pickup_lon > -75 and pickup_lon < -73 and \
           dropoff_lat > 39 and dropoff_lat < 42 and \
           pickup_lat > 39 and pickup_lat < 42:
            outfile.writerow(row)
        else:
            skipped += 1
        total += 1
    print(f'Skipped {skipped} rows out of {total}')
if __name__ == '__main__':
    main()
