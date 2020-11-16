//
// Created by libaoyu on 2020/11/8.
//
#include <iostream>
#include "utils.h"
#include "KDTree.h"
#include "OCTree.h"

int main() {
	using namespace nn_trees;
	// parameter equal with python code from lesson2code
	const static size_type   LeafSize        = 32;
	const static size_type   KNNK            = 8;
	const static scalar_type min_oct_size    = 0.0001;
	const static scalar_type build_test_time = 100;

	auto          points = generate_points("../000000.txt");
	Tictoc        build_kdtree, build_octree, kdtree_search_knn, octree_search_knn, bf_search_knn;
	KDTreeManager kdtree_manager(&points, LeafSize);
	build_kdtree.tic();
	for (int i = 0; i < build_test_time; ++i) {
		KDTreeManager testbuild(&points, LeafSize);
	}
	build_kdtree.toc();

	OCTreeManager octree_manager(&points, LeafSize, min_oct_size);
	build_octree.tic();
	for (int i = 0; i < build_test_time; ++i) {
		OCTreeManager testbuild(&points, LeafSize, min_oct_size);
	}
	build_octree.toc();

	kdtree_search_knn.tic();
	for (int i = 0; i < points.size(); ++i) {
		kdtree_manager.perform_knn_search(points[i], KNNK);
	}
	kdtree_search_knn.toc();

	octree_search_knn.tic();
	for (int i = 0; i < points.size(); ++i) {
		octree_manager.perform_knn_search(points[i], KNNK);
	}
	octree_search_knn.toc();

	printf("kdtree: build time: %f ms, query time: %lu ms\n", (double) build_kdtree.get_last_cnt_ms() / build_test_time,
	       kdtree_search_knn.get_last_cnt_ms());
	printf("octree: build time: %f ms, query time: %lu ms\n", (double) build_octree.get_last_cnt_ms() / build_test_time,
	       octree_search_knn.get_last_cnt_ms());
	return 0;
}