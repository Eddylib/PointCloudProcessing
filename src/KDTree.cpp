//
// Created by libaoyu on 2020/11/3.
//

#include <map>
#include "KDTree.h"

namespace nn_trees {
    scalar_type
    get_split_point(const points_3d &all_points, size_type axis, const indices_type &indices, index_type idx_start,
                    index_type idx_end) {
        // 根据点的坐标集合，对axis维度进行划分，以均值法进行划分
        scalar_type sum = 0;
        for (auto i = idx_start; i < idx_end; i++) {
            sum += all_points[indices[i]][axis];
        }
        return sum / indices.size();
    }

    size_type get_split_dim(const kdtree_3d_node_ptr &node) {
        // 划分维度为x->y->z循环
        return (node->get_axis() + 1) % node->K;
    }

    void split_node(const points_3d &all_points,
                    indices_type &all_indices,
                    index_type idx_start,
                    index_type idx_end,
                    index_type &less_start,
                    index_type &less_end,
                    index_type &greater_start,
                    index_type &greater_end,
                    size_type axis,
                    scalar_type split_point) {
        // 一个点的话根据点的大小来处理
        less_start = idx_start;
        greater_end = idx_end;

        index_type left = idx_start, right = idx_end - 1;
        while (left < right) {
            while (left < right &&all_points[all_indices[left]][axis] <= split_point) {
                left++;
            }
            while (right > left &&all_points[all_indices[right]][axis] > split_point) {
                right--;
            }
            if (left < right) {
                std::swap(all_indices[left], all_indices[right]);
            } else {
                break;
            }
        }
        // 结束时，left指向第一个大于split_point的点， 或者是 idx_end
        less_end = left;
        // 结束时，right指向第一个小于split_point的点， 或者是 idx_start
        greater_start = right;
    }


    kdtree_3d_node_ptr
    kdtree_recursive_build(kdtree_3d_node_ptr &root, const points_3d &all_points, indices_type &indices,
                           size_type axis, size_type leaf_size, index_type idx_start, index_type idx_end) {
//		static size_type cnt = 0;
        if (root == nullptr) {
            root = kdtree_3d_node_type::create_node(axis, 0.0, nullptr, nullptr, idx_start, idx_end);
//			cnt += indices.size();
        }

        // 先把indices给进来，再把indices分下去
        if (indices.size() > leaf_size) {
            // need split
            scalar_type split_point = get_split_point(all_points, root->get_axis(),
                                                      indices, root->idx_start(), root->idx_end());
            root->split_value() = split_point;

            index_type left_start = 0, left_end = 0, right_start, right_end;
            split_node(all_points, indices, idx_start, idx_end, left_start, left_end, right_start, right_end, axis, split_point);
            size_type new_split_dim = get_split_dim(root);
            if (left_start<left_end) {
                kdtree_recursive_build(root->left(), all_points, indices, new_split_dim, leaf_size, left_start, left_end);
            }
            if (right_start<right_end) {
                kdtree_recursive_build(root->right(), all_points, indices, new_split_dim, leaf_size, right_start, right_end);
            }
        }

//		printf("final_indices %d\n", cnt);
        return root;
    }

    kdtree_3d_node_ptr build_tree(const points_3d &points, size_type leaf_size) {
        kdtree_3d_node_ptr ret = nullptr;
        indices_type indices;
        indices.resize(points.size());
        for (size_type i = 0; i < points.size(); ++i) {
            indices[i] = i;
        }
        kdtree_recursive_build(ret, points, indices, 0, leaf_size, 0, indices.size());
        return ret;
    }


    void knn_search_kdtree_recusive(const kdtree_3d_node_ptr &root, const points_3d &all_points,
                                    const indices_type &all_indices,
                                    const point_3d &query_point, ResultManager &result_manager) {
        if (root->is_leaf()) {
            // 对于所有点， 看是否符合条件加入结果集
            for (auto idx = root->idx_start(); idx < root->idx_end(); idx++) {
                auto distance = query_point.distance_with(all_points[all_indices[idx]]);
                result_manager.update_result(idx, distance);
            }
            return;
        }
        scalar_type queue_axis_value = query_point[root->get_axis()];
        scalar_type root_axis_value = root->split_value();
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
    knn_search_kdtree(const kdtree_3d_node_ptr &root,
                      const points_3d &all_points, const point_3d &query_point,
                      int result_size) {
        ResultManager result_manager(result_size);

        knn_search_kdtree_recusive(root, all_points, allindices, query_point, result_manager);
        return result_manager;
    }

    ResultManager knn_search_bf(const points_3d &all_points, const point_3d &query_point, int result_size) {
        ResultManager result_manager(result_size);
        for (size_type i = 0; i < all_points.size(); ++i) {
            auto distance = query_point.distance_with(all_points[i]);
            result_manager.update_result(i, distance);
        }
        return result_manager;
    }

    void kdtree_traverse(kdtree_3d_node_ptr &root, indices_type &test_data) {
        static int leaf_size = 0;
        if (root->is_leaf()) {
            leaf_size += root->get_indices().size();
            for (int i : root->get_indices()) {
                test_data[i] += 1;
            }
        } else {
            if (root->left()) {
                kdtree_traverse(root->left(), test_data);
            }
            if (root->right()) {
                kdtree_traverse(root->right(), test_data);
            }
        }
        printf("leaf_size: %d\n", leaf_size);
    }

    kdtree_3d_node_ptr KDTreeManager::build_tree() {
        kdtree_3d_node_ptr ret = nullptr;
        indices_.resize(points_->size());
        for (size_type i = 0; i < points_->size(); ++i) {
            indices_[i] = i;
        }
        kdtree_recursive_build(ret, *points_, indices_, 0, leaf_size_, 0, indices_.size());
        return nn_trees::kdtree_3d_node_ptr();
    }
}
