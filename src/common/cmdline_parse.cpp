#include <unistd.h>

void parse(int argc, char **argv, int *n_pts, int *min_points, float *epsilon,
           bool *print) {
    // set defaults
    *n_pts = 10000;
    *min_points = 30;
    *epsilon = 0.004;
    *print = false;

    char opt;
    while ((opt = getopt(argc, argv, "n:m:e:p")) != -1) {
        switch (opt) {
        case 'n':
            *n_pts = atoi(optarg);
            break;
        case 'm':
            *min_points = atoi(optarg);
            break;
        case 'e':
            *epsilon = atof(optarg);
            break;
        case 'p':
            *print = true;
            break;
        }
    }
}
