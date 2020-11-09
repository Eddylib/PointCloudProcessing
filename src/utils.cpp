//
// Created by libaoyu on 2020/11/3.
//

#include <iostream>
#include <random>
#include <fstream>
#include "utils.h"

namespace nn_trees {
	const static auto random_seed = 122341234;


	points_type generate_points(const std::string& path) {
		points_type ret;

		using namespace std;
		ifstream    ifs(path);
		scalar_type x, y, z;
		while (ifs >> x >> y >> z) {
			point_type p {x, y, z};
			ret.emplace_back(p);
		}
		return ret;
	}

	points_type generate_points(size_type num) {
		std::default_random_engine       generator(random_seed);
		std::normal_distribution<double> distribution(0, 10);
		points_type                      ret;
		ret.resize(num);
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < ret[i].size(); ++j) {
				ret[i][j] = distribution(generator);
			}
		}
		return ret;
	}

	void print_points(const points_type& points) {
		using namespace std;
		for (const auto& point : points) {
			cout << point[0] << "," << point[1] << "," << point[2] << endl;
		}
	}
}
