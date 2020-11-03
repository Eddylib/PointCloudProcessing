//
// Created by libaoyu on 2020/11/3.
//

#include <iostream>
#include <random>
#include "utils.h"
namespace nn_trees{
	const static auto random_seed = 122341234;
	points_3d generate_points(size_type num){
		std::default_random_engine       generator(random_seed);
		std::normal_distribution<double> distribution(0, 10);
		points_3d                        ret;
		ret.resize(num);
		for (int i = 0; i < num; ++i) {
			for (int j = 0; j < ret[i].size(); ++j) {
				ret[i][j] = distribution(generator);
			}
		}
		return ret;
	}
	void print_points(const points_3d &points){
		using namespace std;
		for (const auto & point : points) {
			cout<<point[0]<<","<<point[1]<<","<<point[2]<<endl;
		}
	}
}
