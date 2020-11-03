//
// Created by libaoyu on 2020/11/3.
//

#include <map>
#include "KDTree.h"

namespace nn_trees {
	scalar_type get_split_point(const points_3d& all_points, size_type axis, const indices_type& indices) {
		// 根据点的坐标集合，对axis维度进行划分，以均值法进行划分
		scalar_type sum = 0;
		for (int indice : indices) {
			sum += all_points[indice][axis];
		}
		return sum / indices.size();
	}

	size_type get_split_dim(const kdtree_3d_node_ptr& node) {
		// 划分维度为x->y->z循环
		return (node->get_axis() + 1) % node->K;
	}

	void split_node(const points_3d& all_points,
	                const indices_type& input_indices,
	                indices_type& less,
	                indices_type& greater,
	                size_type axis,
	                scalar_type split_point) {
		greater.clear();
		less.clear();
		for (int input_indice : input_indices) {
			if (all_points[input_indice][axis] <= split_point) {
				less.push_back(input_indice);
			} else {
				greater.push_back(input_indice);
			}
		}
	}


	kdtree_3d_node_ptr
	kdtree_recursive_build(kdtree_3d_node_ptr &root, const points_3d &all_points, const indices_type &indices,
                           size_type axis, size_type leaf_size, index_type idx_start, index_type idx_end) {
//		static size_type cnt = 0;
		if (root == nullptr) {
			root = kdtree_3d_node_type::create_node(axis, 0.0, nullptr, nullptr, indices);
//			cnt += indices.size();
		}

		// 先把indices给进来，再把indices分下去
		if (indices.size() > leaf_size) {
			// need split
			scalar_type split_point = get_split_point(all_points, root->get_axis(), root->get_indices());
			root->split_value() = split_point;

			indices_type left_indices, right_indices;
			split_node(all_points, root->get_indices(), left_indices, right_indices, root->get_axis(), split_point);
			root->get_indices().clear();
//			cnt -= root->get_indices().size();
//			printf("all %d, left %d, right %d, split_point %f, split_axis %d\n", root->get_indices().size(),
//			       left_indices.size(), right_indices.size(), split_point, root->get_axis());
			size_type new_split_dim = get_split_dim(root);
			if(!left_indices.empty()){
                kdtree_recursive_build(root->left(), all_points, left_indices, new_split_dim, leaf_size, 0, 0);
			}
			if(!right_indices.empty()){
                kdtree_recursive_build(root->right(), all_points, right_indices, new_split_dim, leaf_size, 0, 0);
			}
		}

//		printf("final_indices %d\n", cnt);
		return root;
	}

	kdtree_3d_node_ptr build_tree(const points_3d& points, size_type leaf_size) {
		kdtree_3d_node_ptr ret = nullptr;
		indices_type       indices;
		indices.resize(points.size());
		for (size_type i = 0; i < points.size(); ++i) {
			indices[i] = i;
		}
        kdtree_recursive_build(ret, points, indices, 0, leaf_size, 0, indices.size());
		return ret;
	}




	void knn_search_kdtree_recusive(const kdtree_3d_node_ptr& root,
	                                const points_3d& all_points, const point_3d& query_point,
	                                ResultManager& result_manager) {
		if (root->is_leaf()) {
			// 对于所有点， 看是否符合条件加入结果集
			for (auto idx : root->get_indices()) {
				auto distance = query_point.distance_with(all_points[idx]);
				result_manager.update_result(idx, distance);
			}
			return;
		}
		scalar_type queue_axis_value = query_point[root->get_axis()];
		scalar_type root_axis_value  = root->split_value();
		if (queue_axis_value < root->split_value()) {
			knn_search_kdtree_recusive(root->left(), all_points, query_point, result_manager);
			// 有必要找另一边则找另一边
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive(root->right(), all_points, query_point, result_manager);
			}
		} else {
			knn_search_kdtree_recusive(root->right(), all_points, query_point, result_manager);
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive(root->left(), all_points, query_point, result_manager);
			}
		}
	}

	ResultManager
	knn_search_kdtree(const kdtree_3d_node_ptr& root,
	                  const points_3d& all_points, const point_3d& query_point,
	                  int result_size) {
		ResultManager result_manager(result_size);

		knn_search_kdtree_recusive(root, all_points, query_point, result_manager);
		return result_manager;
	}

	ResultManager knn_search_bf(const points_3d& all_points, const point_3d& query_point, int result_size) {
        ResultManager result_manager(result_size);
        for (size_type i = 0; i < all_points.size(); ++i) {
            auto distance = query_point.distance_with(all_points[i]);
            result_manager.update_result(i,distance);
        }
		return result_manager;
	}

    void kdtree_traverse(kdtree_3d_node_ptr &root, indices_type &test_data) {
	    static int leaf_size = 0;
	    if(root->is_leaf()){
	        leaf_size += root->get_indices().size();
            for (int i : root->get_indices()) {
                test_data[i] += 1;
            }
	    }else{
	        if(root->left()){
                kdtree_traverse(root->left(), test_data);
	        }
	        if(root->right()){
                kdtree_traverse(root->right(), test_data);
	        }
	    }
	    printf("leaf_size: %d\n", leaf_size);
	}
}
