//
// Created by libaoyu on 2020/11/3.
//

#include <map>
#include "KDTree.h"

namespace nn_trees {
	size_type get_split_dim(const kdtree_node_ptr& node) {
		// 划分维度为x->y->z循环
		return (node->get_axis() + 1) % node->K;
	}

	KNNResultManager
	knn_search_kdtree(const kdtree_node_ptr& root,
	                  const points_type& all_points, const point_type& query_point,
	                  int result_size) {
		KNNResultManager result_manager(result_size);

//        knn_search_kdtree_recusive(root, all_points, allindices, query_point, result_manager);
		return result_manager;
	}

	KNNResultManager knn_search_knn_bf(const points_type& all_points, const point_type& query_point, int result_size) {
		KNNResultManager result_manager(result_size);
		for (size_type   i = 0; i < all_points.size(); ++i) {
			auto distance = query_point.distance_with(all_points[i]);
			result_manager.update_result(i, distance);
		}
		return result_manager;
	}
	RadisusResultManager knn_search_radius_bf(const points_type& all_points, const point_type& query_point, scalar_type radius) {
		RadisusResultManager result_manager(radius);
		for (size_type   i = 0; i < all_points.size(); ++i) {
			auto distance = query_point.distance_with(all_points[i]);
			result_manager.update_result(i, distance);
		}
		return result_manager;
	}
	kdtree_node_ptr KDTreeManager::build_tree_() {
		kdtree_node_ptr ret = nullptr;
		indices_.resize(points_->size());
		for (size_type i = 0; i < points_->size(); ++i) {
			indices_[i] = i;
		}
		kdtree_recursive_build_(ret, 0, 0, indices_.size());
		return ret;
	}

	void KDTreeManager::kdtree_recursive_build_(kdtree_node_ptr& root, size_type axis, index_type idx_start,
	                                            index_type idx_end) {
		assert(root == nullptr);
		if (root == nullptr) {
			root = kdtree_node_type::create_node(axis, 0.0, nullptr, nullptr, idx_start, idx_end);
		}

		// 先把indices给进来，再把indices分下去
		size_type indices_size = idx_end - idx_start;
		if (indices_size > leaf_size_) {
			// need split
			scalar_type split_point = get_split_point_(axis, idx_start, idx_end);
			root->split_value() = split_point;

			index_type left_start = 0, left_end = 0, right_start = 0, right_end=0;
			split_node(idx_start, idx_end, left_start, left_end, right_start, right_end, axis,
			           split_point);

//			printf("before: %d-->%d, after: left %d-->%d, right %d-->%d, num: %d-->%d+%d\n", idx_start, idx_end,
//			       left_start, left_end, right_start, right_end, idx_end - idx_start, left_end - left_start,
//			       right_end - right_start);

			size_type new_split_dim = get_split_dim(root);
			if (left_start < left_end) {
				kdtree_recursive_build_(root->left(), new_split_dim, left_start, left_end);
			}
			if (right_start < right_end) {
				kdtree_recursive_build_(root->right(), new_split_dim, right_start, right_end);
			}
		}

	}

	scalar_type KDTreeManager::get_split_point_(size_type axis, index_type idx_start, index_type idx_end) const {
		// 根据点的坐标集合，对axis维度进行划分，以均值法进行划分
		scalar_type sum = 0;
		for (auto   i   = idx_start; i < idx_end; i++) {
			sum += (*points_)[indices_[i]][axis];
		}
		return sum / (idx_end - idx_start);
	}

	RadisusResultManager KDTreeManager::perform_radius_search(point_type const& query, scalar_type radius) {
		RadisusResultManager result_manager(radius);
		knn_search_kdtree_recusive_(root_, query, result_manager);
		return result_manager;
	}

	KNNResultManager KDTreeManager::perform_knn_search(point_type const& query, size_type k_nn) {
		// 没有静态变量，result为当前栈内变量，可重入
		KNNResultManager result_manager(k_nn);
		knn_search_kdtree_recusive_(root_, query, result_manager);
		return result_manager;
	}

	void KDTreeManager::knn_search_kdtree_recusive_(const kdtree_node_ptr& root, const point_type& query_point,
	                                                ResultManagerBase& result_manager) {
		if (root->is_leaf()) {
			// 对于所有点， 看是否符合条件加入结果集
			for (auto idx = root->idx_start(); idx < root->idx_end(); idx++) {
				auto distance = query_point.distance_with((*points_)[indices_[idx]]);
				result_manager.update_result(indices_[idx], distance);
			}
			return;
		}
		scalar_type queue_axis_value = query_point[root->get_axis()];
		scalar_type root_axis_value  = root->split_value();
		if (queue_axis_value < root->split_value()) {
			knn_search_kdtree_recusive_(root->left(), query_point, result_manager);
			// 有必要找另一边则找另一边
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive_(root->right(), query_point, result_manager);
			}
		} else {
			knn_search_kdtree_recusive_(root->right(), query_point, result_manager);
			if (std::abs(queue_axis_value - root_axis_value) < result_manager.get_max_distance()) {
				knn_search_kdtree_recusive_(root->left(), query_point, result_manager);
			}

		}

	}

	void
	KDTreeManager::split_node(index_type idx_start, index_type idx_end, index_type& less_start, index_type& less_end,
	                          index_type& greater_start, index_type& greater_end, size_type axis,
	                          scalar_type split_point) {
		// 一个点的话根据点的大小来处理
		less_start  = idx_start;
		greater_end = idx_end;
		index_type left = idx_start, right = idx_end - 1;
		while (left < right) {
			while (left < right && (*points_)[indices_[left]][axis] <= split_point) {
				left++;
			}
			while (right > left && (*points_)[indices_[right]][axis] > split_point) {
				right--;
			}
			if (left < right) {
				std::swap(indices_[left], indices_[right]);
			} else {
				break;
			}
		}
		// 结束时，left指向第一个大于split_point的点， 或者是 idx_end
		less_end      = left;
		// 结束时，right指向第一个小于split_point的点， 或者是 idx_start
		greater_start = right;
	}

}