//
// Created by libaoyu on 2020/11/5.
//

#ifndef NNTREES_OCTREE_H
#define NNTREES_OCTREE_H

#include <memory>
#include <utility>
#include <queue>
#include "common_types.h"

namespace nn_trees {
	class OCTreeNode;

	using oc_tree_point_type = Point<scalar_type, 3>;
	using oc_tree_node_ptr = std::shared_ptr<OCTreeNode>;

	class OCTreeNode {
	public:
		OCTreeNode(const oc_tree_point_type& center, scalar_type oct_size) : center_(center), oct_size_(oct_size) {}

		static oc_tree_node_ptr create_octree_node(const oc_tree_point_type& center, scalar_type oct_size) {
			return std::make_shared<OCTreeNode>(center, oct_size);
		};

		bool is_leaf()const {
			return !point_indices.empty();
		}

		size_type indices_size() const {
			return point_indices.size();
		}

		void set_indices_num(size_type num){
			num_indices = num;
		}
		size_type get_indices(index_type i){
			return point_indices[i];
		}

		void set_points_indices(const indices_type& indices) {
			point_indices = indices;
		}

		oc_tree_node_ptr& child(int i) { return childs_[i]; }
		const oc_tree_node_ptr& child(int i) const{ return childs_[i]; }
		oc_tree_point_type &center(){return center_;}
		const oc_tree_point_type &center()const {return center_;}
		scalar_type oct_size()const{return oct_size_;}
	private:
		oc_tree_node_ptr   childs_[8];
		oc_tree_point_type center_;
		scalar_type        oct_size_;
		indices_type       point_indices;
		size_type          num_indices = 0;
	};

	using oc_tree_node_type = OCTreeNode;

	class OCTreeManager {
	public:
		OCTreeManager(points_type* points, size_type leaf_size, scalar_type min_oct) :
				points_(points), leaf_size_(leaf_size), min_oct_(min_oct), root_(nullptr) {
			build_tree();
//			traverse_check();
		}

		KNNResultManager perform_knn_search(const oc_tree_point_type & query, size_type k_nn);
		RadisusResultManager perform_radius_search(const oc_tree_point_type & query, scalar_type radius);
		RadisusResultManager perform_radius_search_fast(const oc_tree_point_type & query, scalar_type radius);
	private:
		void build_tree();

		void build_tree_recursive(nn_trees::oc_tree_node_ptr& root,
		                          nn_trees::oc_tree_point_type const& center,
		                          nn_trees::scalar_type oct_size,
		                          const indices_type& indices);

		void traverse_check();

		void traverse_recursive(const oc_tree_node_ptr& root, size_type& leaf_cnt);

		bool search_recursive(const oc_tree_node_ptr &root, oc_tree_point_type const& query, ResultManagerBase& result_manager);

		points_type* points_;
		size_type        leaf_size_;
		scalar_type      min_oct_;
		oc_tree_node_ptr root_;
	};
}


#endif //NNTREES_OCTREE_H
