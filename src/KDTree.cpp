//
// Created by libaoyu on 2020/11/3.
//

#include <map>
#include "KDTree.h"

namespace nn_trees {
	size_type get_split_dim(const kdtree_3d_node_ptr& node) {
		// 划分维度为x->y->z循环
		return (node->get_axis() + 1) % node->K;
	}

	void split_node(const points_3d& all_points,
	                indices_type& all_indices,
	                index_type idx_start,
	                index_type idx_end,
	                index_type& less_start,
	                index_type& less_end,
	                index_type& greater_start,
	                index_type& greater_end,
	                size_type axis,
	                scalar_type split_point) {
		// 一个点的话根据点的大小来处理
		less_start  = idx_start;
		greater_end = idx_end;
		index_type left = idx_start, right = idx_end - 1;
		while (left < right) {
			while (left < right && all_points[all_indices[left]][axis] <= split_point) {
				left++;
			}
			while (right > left && all_points[all_indices[right]][axis] > split_point) {
				right--;
			}
			if (left < right) {
				std::swap(all_indices[left], all_indices[right]);
			} else {
				break;
			}
		}
		// 结束时，left指向第一个大于split_point的点， 或者是 idx_end
		less_end      = left;
		// 结束时，right指向第一个小于split_point的点， 或者是 idx_start
		greater_start = right;
	}




	void knn_search_kdtree_recusive(const kdtree_3d_node_ptr& root, const points_3d& all_points,
	                                const indices_type& all_indices,
	                                const point_3d& query_point, ResultManager& result_manager) {
		if (root->is_leaf()) {
			// 对于所有点， 看是否符合条件加入结果集
			for (auto idx = root->idx_start(); idx < root->idx_end(); idx++) {
				auto distance = query_point.distance_with(all_points[all_indices[idx]]);
				result_manager.update_result(idx, distance);
			}
			return;
		}
		scalar_type queue_axis_value = query_point[root->get_axis()];
		scalar_type root_axis_value  = root->split_value();
		if (queue_axis_value < root->split_value()) {
			knn_search_kdtree_recusive(root->left(), all_points, all_indices, query_point, result_manager);
			// 有必要找另一边则找另一边
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive(root->right(), all_points, all_indices, query_point, result_manager);
			}
		} else {
			knn_search_kdtree_recusive(root->right(), all_points, all_indices, query_point, result_manager);
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive(root->left(), all_points, all_indices, query_point, result_manager);
			}
		}
	}

	ResultManager
	knn_search_kdtree(const kdtree_3d_node_ptr& root,
	                  const points_3d& all_points, const point_3d& query_point,
	                  int result_size) {
		ResultManager result_manager(result_size);

//        knn_search_kdtree_recusive(root, all_points, allindices, query_point, result_manager);
		return result_manager;
	}

	ResultManager knn_search_bf(const points_3d& all_points, const point_3d& query_point, int result_size) {
		ResultManager  result_manager(result_size);
		for (size_type i = 0; i < all_points.size(); ++i) {
			auto distance = query_point.distance_with(all_points[i]);
			result_manager.update_result(i, distance);
		}
		return result_manager;
	}

//    void kdtree_traverse(kdtree_3d_node_ptr &root, indices_type &test_data) {
//        static int leaf_size = 0;
//        if (root->is_leaf()) {
//            leaf_size += root->get_indices().size();
//            for (int i : root->get_indices()) {
//                test_data[i] += 1;
//            }
//        } else {
//            if (root->left()) {
//                kdtree_traverse(root->left(), test_data);
//            }
//            if (root->right()) {
//                kdtree_traverse(root->right(), test_data);
//            }
//        }
//        printf("leaf_size: %d\n", leaf_size);
//    }

	kdtree_3d_node_ptr KDTreeManager::build_tree() {
		kdtree_3d_node_ptr ret = nullptr;
		indices_.resize(points_->size());
		for (size_type i = 0; i < points_->size(); ++i) {
			indices_[i] = i;
		}
		kdtree_recursive_build(ret, 0, 0, indices_.size());
		return ret;
	}

	void
	KDTreeManager::kdtree_recursive_build(kdtree_3d_node_ptr& root, size_type axis, index_type idx_start,
	                                      index_type idx_end) {
		if (root == nullptr) {
			root = kdtree_3d_node_type::create_node(axis, 0.0, nullptr, nullptr, idx_start, idx_end);
		}

		// 先把indices给进来，再把indices分下去
		size_type indices_size = idx_end - idx_start;
		if (indices_size > leaf_size_) {
			// need split
			scalar_type split_point = get_split_point(axis,idx_start, idx_end);
			root->split_value() = split_point;

			index_type left_start = 0, left_end = 0, right_start, right_end;
			split_node(*points_, indices_, idx_start, idx_end, left_start, left_end, right_start, right_end, axis,
			           split_point);

//			printf("before: %d-->%d, after: left %d-->%d, right %d-->%d, num: %d-->%d+%d\n", idx_start, idx_end,
//			       left_start, left_end, right_start, right_end, idx_end - idx_start, left_end - left_start,
//			       right_end - right_start);

			size_type new_split_dim = get_split_dim(root);
			if (left_start < left_end) {
				kdtree_recursive_build(root->left(), new_split_dim, left_start, left_end);
			}
			if (right_start < right_end) {
				kdtree_recursive_build(root->right(), new_split_dim, right_start, right_end);
			}
		}

	}

	scalar_type KDTreeManager::get_split_point(size_type axis, index_type idx_start, index_type idx_end) const {
		// 根据点的坐标集合，对axis维度进行划分，以均值法进行划分
		scalar_type sum = 0;
		for (auto   i   = idx_start; i < idx_end; i++) {
			sum += (*points_)[indices_[i]][axis];
		}
		return sum / (idx_end-idx_start);
	}
}
