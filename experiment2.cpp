#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cmath>
#include <fstream>

struct Point {
    double x;
    double y;
    bool trueBlue;
    bool labelBlue;
};

bool isInsideTriangle(const Point& p) {
    return p.y >= 3.0 &&
           p.x <= 7.0 &&
           p.y <= p.x;
}

double squaredDistance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;

    return dx * dx + dy * dy;
}

// Predict using training labels only.
bool predictBlue(const std::vector<Point>& training,
                 const Point& query,
                 int k) {
    if (k <= 0 ||
        k % 2 == 0 ||
        static_cast<std::size_t>(k) > training.size()) {
        throw std::invalid_argument(
            "k must be positive, odd, and <= training size"
        );
    }

    std::vector<std::pair<double, std::size_t>> neighbors;
    neighbors.reserve(training.size());

    for (std::size_t i = 0; i < training.size(); ++i) {
        neighbors.push_back({
            squaredDistance(query, training[i]),
            i
        });
    }

    // Sort by distance; break distance ties using the point index.
    std::sort(neighbors.begin(), neighbors.end());

    int blueVotes = 0;

    for (int i = 0; i < k; ++i) {
        const Point& neighbor = training[neighbors[i].second];

        if (neighbor.labelBlue) {
            ++blueVotes;
        }
    }

    return blueVotes > k / 2;
}

// Evaluate on new test points without modifying the training set.
int evaluate(const std::vector<Point>& training,
             int k,
             int testCount,
             unsigned int testSeed) {
    if (testCount <= 0) {
        throw std::invalid_argument("testCount must be positive");
    }

    std::mt19937 testGenerator(testSeed);
    std::uniform_real_distribution<double> coordinate(0.0, 10.0);

    int errors = 0;

    for (int i = 0; i < testCount; ++i) {
        Point testPoint{
            coordinate(testGenerator),
            coordinate(testGenerator),
            false,
            false
        };

        bool correctLabel = isInsideTriangle(testPoint);
        bool prediction = predictBlue(training, testPoint, k);

        if (prediction != correctLabel) {
            ++errors;
        }
    }

    return errors;
}

std::vector<Point> generateTraining(int n,
                                    double f,
                                    unsigned int seed) {
    if (n <= 0 || !(f >= 0.0 && f <= 1.0)) {
        throw std::invalid_argument(
            "n must be positive and f must be between 0 and 1"
        );
    }

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> coordinate(0.0, 10.0);
    std::bernoulli_distribution flipLabel(f);

    std::vector<Point> points;
    points.reserve(n);

    // Generate coordinates first.
    for (int i = 0; i < n; ++i) {
        Point p{
            coordinate(generator),
            coordinate(generator),
            false,
            false
        };

        p.trueBlue = isInsideTriangle(p);
        p.labelBlue = p.trueBlue;

        points.push_back(p);
    }

    // Then flip training labels.
    for (Point& p : points) {
        if (flipLabel(generator)) {
            p.labelBlue = !p.labelBlue;
        }
    }

    return points;
}

int main() {
    const int n = 1000;
    const int k = 7;
    const int testCount = 10000;
    const int repeats = 30;

    std::ofstream rawFile("experiment2_raw.csv");
    std::ofstream summaryFile("experiment2_summary.csv");

    if (!rawFile || !summaryFile) {
        std::cerr << "Could not open output files.\n";
        return 1;
    }

    rawFile
        << "n,k,f,test_count,run,training_seed,test_seed,errors\n";

    summaryFile
        << "n,k,f,test_count,repeats,mean_errors,sample_sd\n";

    rawFile << std::setprecision(10);
    summaryFile << std::fixed << std::setprecision(6);
    std::cout << std::fixed << std::setprecision(2);

    for (int setting = 0; setting <= 6; ++setting) {
        double f = setting / 20.0;
        std::vector<int> errorCounts;

        for (int run = 0; run < repeats; ++run) {
            unsigned int runId =
                static_cast<unsigned int>(setting * 1000 + run);

            unsigned int trainingSeed = 100042u + runId;
            unsigned int testSeed = 200045u + runId;

            std::vector<Point> points =
                generateTraining(n, f, trainingSeed);

            int errors =
                evaluate(points, k, testCount, testSeed);

            errorCounts.push_back(errors);

            rawFile << n << ','
                    << k << ','
                    << f << ','
                    << testCount << ','
                    << run + 1 << ','
                    << trainingSeed << ','
                    << testSeed << ','
                    << errors << '\n';
        }

        double sum = 0.0;

        for (int errors : errorCounts) {
            sum += errors;
        }

        double mean = sum / errorCounts.size();
        double squaredDifferences = 0.0;

        for (int errors : errorCounts) {
            double difference = errors - mean;
            squaredDifferences += difference * difference;
        }

        double sd = std::sqrt(
            squaredDifferences / (errorCounts.size() - 1)
        );

        summaryFile << n << ','
                    << k << ','
                    << f << ','
                    << testCount << ','
                    << repeats << ','
                    << mean << ','
                    << sd << '\n';

        std::cout << "f = " << f
                  << " | Mean errors = " << mean
                  << " | Sample SD = " << sd << std::endl;
    }

    rawFile.close();
    summaryFile.close();

    if (!rawFile || !summaryFile) {
        std::cerr << "Failed to save results.\n";
        return 1;
    }

    std::cout << "\nSaved: experiment2_raw.csv\n";
    std::cout << "Saved: experiment2_summary.csv\n";

    return 0;
}