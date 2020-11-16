//
// Created by libaoyu on 2020/11/3.
//

#ifndef NNTREES_KDTREE_H
#define NNTREES_KDTREE_H

#include <memory>
#include <utility>
#include "common_types.h"

namespace nn_trees {

	template<int Dimension>
	class KDTreeNode {
	public:
		using node_pointer_type = std::shared_ptr<KDTreeNode>;
		using indices_type = std::vector<index_type>;

		constexpr static size_type K = Dimension;

		KDTreeNode() = default;

		KDTreeNode(
				size_type axis,
				scalar_type value,
				node_pointer_type left,
				node_pointer_type right,
				index_type idx_start, index_type index_end
		) : axis_(axis),
		    value_(value),
		    left_(std::move(left)),
		    right_(std::move(right)), start_(idx_start), end_(index_end) {
			assert(axis_ < K);
		}

		static node_pointer_type create_node(
				size_type axis,
				scalar_type value,
				const node_pointer_type& left,
				const node_pointer_type& right,
				index_type idx_start, index_type idx_end
		) {
			return std::make_shared<KDTreeNode>(axis, value, left, right, idx_start, idx_end);
		}

		bool_type is_leaf() const {
			return left_ == nullptr && right_ == nullptr;
		}

		node_pointer_type& left() { return left_; }

		node_pointer_type& right() { return right_; }

		scalar_type& value() { return value_; }

		scalar_type& split_value() {
			return value_;
		}

		const node_pointer_type& left() const { return left_; }

		const node_pointer_type& right() const { return right_; }

		size_type get_axis() const { return axis_; }

		index_type idx_start() const { return start_; }

		index_type idx_end() const { return end_; }

	private:
		size_type         axis_;
		scalar_type       value_;
		node_pointer_type left_;
		node_pointer_type right_;
		// 避免内存占用，只用start和end来表示下标
		index_type        start_;
		index_type        end_;
	};

	using kdtree_node_type = KDTreeNode<point_type::size()>;
	using kdtree_node_ptr = kdtree_node_type::node_pointer_type;


	class KDTreeManager {
	public:
		KDTreeManager(points_type* points, size_type leaf_size) : points_(points), leaf_size_(leaf_size) {
			root_ = build_tree_();
		}

		KNNResultManager perform_knn_search(const point_type& query, size_type k_nn);
		RadisusResultManager perform_radius_search(const point_type& query, scalar_type radius);

	private:
		void knn_search_kdtree_recusive_(const kdtree_node_ptr& root, const point_type& query_point,
		                                 ResultManagerBase& result_manager);

		void kdtree_recursive_build_(kdtree_node_ptr& root, size_type axis, index_type idx_start, index_type idx_end);

		kdtree_node_ptr build_tree_();

		scalar_type get_split_point_(size_type axis, index_type idx_start, index_type idx_end) const;

		void split_node(
		                index_type idx_start,
		                index_type idx_end,
		                index_type& less_start,
		                index_type& less_end,
		                index_type& greater_start,
		                index_type& greater_end,
		                size_type axis,
		                scalar_type split_point);

		indices_type indices_;
		points_type* points_;
		size_type       leaf_size_;
		kdtree_node_ptr root_;
	};

	KNNResultManager
	knn_search_kdtree(const kdtree_node_ptr& root,
	                  const points_type& all_points, const point_type& query_point,
	                  int result_size);

	KNNResultManager knn_search_knn_bf(const points_type& all_points, const point_type& query_point, int result_size);
	RadisusResultManager knn_search_radius_bf(const points_type& all_points, const point_type& query_point, scalar_type radius);

}


#endif //NNTREES_KDTREE_H

