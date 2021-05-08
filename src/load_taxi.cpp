#include "point_set.hpp"
#include "load_taxi.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

// 0-based index of fields in csv
int DROPOFF_LONG_POS = 9;
int DROPOFF_LAT_POS = 10;
int PICKUP_LONG_POS = 5;
int PICKUP_LAT_POS = 6;

PointSet get_n_dropoffs(int n, const char *dataset) {
    /**
     * Load at most n dropoffs from taxi data stored in a file
     * called dataset. Returns a PointSet object. The size of the PointSet
     * is stored in the .size attribute. If dataset is a null pointer, 
     * the default dataset (Jan 2015 Yellow Taxi data) is used
     */
    if (!dataset) {
        dataset = "data/CLEANED_yellow_tripdata_2015-01.csv";
    }
    PointSet ps(n);
    fstream file;
    file.open(dataset, ios::in);
    string line;
    string field;
    int i = 0;
    getline(file, line); // skip header
    while (!file.eof() && i < n) {
        getline(file, line);
        int idx = 0;
        int pos = 0;
        while (idx < DROPOFF_LONG_POS) {
            pos = line.find(",", pos) + 1;
            idx += 1;
        }
        string::size_type s;
        float x = stof(line.substr(pos), &s);
        float y = stof(line.substr(pos + s + 1), NULL);
        ps.set(i, x, y);
        i += 1;
    }
    if (i < n) {
        // User asked for more data points than are available so resize the
        // point set
        ps.resize(i);
    }
    return ps;
}

PointSet get_n_pickups(int n, const char *dataset) {
    /**
     * Load at most n pickups from taxi data stored in a file
     * called dataset. Returns a PointSet object. The size of the PointSet
     * is stored in the .size attribute. If dataset is a null pointer, 
     * the default dataset (Jan 2015 Yellow Taxi data) is used
     */
    if (!dataset) {
        dataset = "data/CLEANED_yellow_tripdata_2015-01.csv";
    }
    PointSet ps(n);
    fstream file;
    file.open(dataset, ios::in);
    string line;
    string field;
    int i = 0;
    getline(file, line); // skip header
    while (!file.eof() && i < n) {
        getline(file, line);
        int idx = 0;
        int pos = 0;
        while (idx < PICKUP_LONG_POS) {
            pos = line.find(",", pos) + 1;
            idx += 1;
        }
        string::size_type s;
        float x = stof(line.substr(pos), &s);
        float y = stof(line.substr(pos + s + 1), NULL);
        ps.set(i, x, y);
        i += 1;
    }
    if (i < n) {
        // User asked for more data points than are available so resize the
        // point set
        ps.resize(i);
    }
    return ps;
}
