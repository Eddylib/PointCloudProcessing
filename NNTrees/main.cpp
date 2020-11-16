#include <iostream>
#include "utils.h"
#include "KDTree.h"
#include "OCTree.h"

int main() {
	using namespace nn_trees;
	const static size_type   LeafSize     = 32;
	const static size_type   KNNK         = 8;
	const static scalar_type Radius       = 1;
	const static scalar_type min_oct_size = 0.0001;

	auto          points = generate_points("../000000.txt");
	KDTreeManager kdtree_manager(&points, LeafSize);
	OCTreeManager octree_manager(&points, LeafSize, min_oct_size);

	for (int i = 0; i < points.size(); ++i) {
		index_type query_idx         = i;
		if(i%1000 == 0){
			printf("query idx %d of %d points\n", query_idx, points.size());
		}
		auto       result_kdtree_knn = kdtree_manager.perform_knn_search(points[query_idx], KNNK);
		auto       result_octree_knn = octree_manager.perform_knn_search(points[query_idx], KNNK);
		auto       result_bf_knn     = knn_search_knn_bf(points, points[query_idx], KNNK);
		if(result_bf_knn.get_result() != result_kdtree_knn.get_result()){
			result_kdtree_knn.show_results();
			result_bf_knn.show_results();
			exit(0);
		}
		if(result_octree_knn.get_result() != result_bf_knn.get_result()){
			result_octree_knn.show_results();
			result_bf_knn.show_results();
			exit(0);
		}
	}


	return 0;
}