#include <iostream>
#include "utils.h"
#include "KDTree.h"

int main() {
    using namespace nn_trees;
    const static size_type LeafSize = 1;
    const static size_type KNNK = 10;
    const static size_type point_num = 100*10000;
    const static index_type query_idx = 100;
	#if 0
    std::vector<kdtree_3d_node_type> kdtreetest;
	std::vector<kdtree_3d_node_ptr> kdtreetest_ptr;
	kdtreetest.resize(point_num);
	kdtreetest_ptr.resize(point_num);
    indices_type indices;
    indices.resize(point_num);
	#endif
    auto points = generate_points(point_num);
    Tictoc build, kdtree_sarch, bf_search;
    build.tic();
    KDTreeManager manager(&points, LeafSize);
    build.toc();

    kdtree_sarch.tic();
	for (int i = 0; i < point_num; ++i) {
		auto resultkdtree = manager.perform_search(points[query_idx], KNNK);
	}
    kdtree_sarch.toc();


    bf_search.tic();
    auto resultbf = knn_search_bf(points, points[query_idx], KNNK);
    bf_search.toc();

//    if (resultbf.get_result() == resultkdtree.get_result()) {
//        printf("anser equal!\n");
//    }
    printf("build time: %lu us, bf_search time %lu us, kd_tree_search %lu us",
           build.get_last_cnt_us(),
           bf_search.get_last_cnt_us(),
           kdtree_sarch.get_last_cnt_us());
    return 0;
}