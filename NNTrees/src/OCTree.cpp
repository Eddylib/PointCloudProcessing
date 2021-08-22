//
// Created by libaoyu on 2020/11/5.
//

#include "OCTree.h"

namespace nn_trees {
	void OCTreeManager::build_tree() {
		assert(points_);

		oc_tree_point_type oct_min {std::numeric_limits<scalar_type>::max(),
		                            std::numeric_limits<scalar_type>::max(),
		                            std::numeric_limits<scalar_type>::max()};
		oc_tree_point_type oct_max {std::numeric_limits<scalar_type>::min(),
		                            std::numeric_limits<scalar_type>::min(),
		                            std::numeric_limits<scalar_type>::min()};
		auto& points = *points_;

		for (auto& point : points) {
			if (oct_min[0] > point[0]) oct_min[0] = point[0];
			if (oct_min[1] > point[1]) oct_min[1] = point[1];
			if (oct_min[2] > point[2]) oct_min[2] = point[2];
			if (oct_max[0] < point[0]) oct_max[0] = point[0];
			if (oct_max[1] < point[1]) oct_max[1] = point[1];
			if (oct_max[2] < point[2]) oct_max[2] = point[2];
		}
		scalar_type oct_size = std::max(oct_max[0] - oct_min[0],
		                                std::max(oct_max[1] - oct_min[1],
		                                         oct_max[2] - oct_min[2])) * 0.5;

		point_type center = {oct_min[0] + oct_size,
		                     oct_min[1] + oct_size,
		                     oct_min[2] + oct_size};

		indices_type indices;
		indices.resize(points_->size());
		for (auto i = 0; i < points_->size(); ++i) {
			indices[i] = i;
		}
		build_tree_recursive(root_, center, oct_size, indices);
	}

	void OCTreeManager::build_tree_recursive(oc_tree_node_ptr& root,
	                                         oc_tree_point_type const& center,
	                                         scalar_type oct_size,
	                                         const indices_type& indices) {
		if (indices.empty()) {
			return;
		}

		if (root == nullptr) {
			root = oc_tree_node_type::create_octree_node(center, oct_size);
		}

		root->set_indices_num(indices.size());

		if (indices.size() <= leaf_size_ || oct_size < min_oct_) {
			root->set_points_indices(indices);
		} else {
			indices_type child_indices[8];
			auto& points = *points_;
			for (int indice : indices) {
				auto& point = points[indice];
				uint8_t oct_code = (point[0] > center[0])
				                   + ((point[1] > center[1]) << 1u)
				                   + ((point[2] > center[2]) << 2u);
				child_indices[oct_code].push_back(indice);
			}
			#if 0
			printf("%zu pointts--> ", indices.size());
			for (int i = 0; i < 8; ++i) {
				printf(" %zu", child_indices[i].size());
			}
			printf("\n");
			#endif

			const static scalar_type factor[] = {-0.5, 0.5};

			for (uint8_t i = 0; i < 8; ++i) {
				point_type  child_center {center[0] + factor[(i & 1u)>0] * oct_size,
				                          center[1] + factor[(i & 2u)>0] * oct_size,
				                          center[2] + factor[(i & 4u)>0] * oct_size};
				scalar_type child_oct_size = oct_size / 2.0;
				build_tree_recursive(root->child(i), child_center, child_oct_size, child_indices[i]);
			}
		}
	}

	void OCTreeManager::traverse_check() {
		size_type leaf_cnt = 0;
		traverse_recursive(root_, leaf_cnt);
		printf("check: %lu points at leaf\n", leaf_cnt);
	}

	void OCTreeManager::traverse_recursive(const oc_tree_node_ptr& root, size_type& leaf_cnt) {
		if (root->is_leaf()) {
			leaf_cnt += root->indices_size();
			return;
		} else {
			for (auto i = 0; i < 8; ++i) {
				if (root->child(i)) {
					traverse_recursive(root->child(i), leaf_cnt);
				}
			}
		}
	}

	RadisusResultManager OCTreeManager::perform_radius_search(const oc_tree_point_type & query, scalar_type radius){
		RadisusResultManager manager(radius);
		search_recursive(root_, query, manager);
		return manager;
	}

	KNNResultManager OCTreeManager::perform_knn_search(oc_tree_point_type const& query, size_type k_nn) {
		KNNResultManager manager(k_nn);
		search_recursive(root_, query, manager);
		return manager;
	}

	bool warst_inside_oct(const oc_tree_node_ptr& root, const oc_tree_point_type& query, const scalar_type& r) {
		// 问题为一个球是否在一个立方体块内
		// 球心与块心的三维的距离都小于块的半边长
		for (int i = 0; i < 3; ++i) {
			if (std::abs(root->center()[i] - query[i]) + r > root->oct_size()) {
				return false;
			}
		}
		return true;
	}

	bool warst_overlap_oct(const oc_tree_node_ptr& root, const oc_tree_point_type& query, const scalar_type& r) {
		std::vector<scalar_type> diff_abs(3, 0);

		for (int i = 0; i < 3; ++i) {
			diff_abs[i] = std::abs(root->center()[i] - query[i]);
		}

		// 1、 一个维度的距离超过半径+块长，则肯定不想交

		auto max_dist = r + root->oct_size();

		for (int   i   = 0; i < 3; ++i) {
			if (diff_abs[i] > max_dist) {
				return false;
			}
		}
		// 2、所有维度差小于半径+块长，两个维度差小于块长，球与立方体面相交
		index_type cnt = 0;

		for (int i = 0; i < 3; ++i) {
			if (diff_abs[i] < root->oct_size()) {
				cnt++;
			}
		}
		if (cnt >= 2)
			return true;

		// 3、立方块的顶点与球相交，即点在球体内， 而边相交可以看做降维的点相交
		scalar_type diffx = std::max(diff_abs[0] - root->oct_size(), static_cast<scalar_type>(0.0));
		scalar_type diffy = std::max(diff_abs[1] - root->oct_size(), static_cast<scalar_type>(0.0));
		scalar_type diffz = std::max(diff_abs[2] - root->oct_size(), static_cast<scalar_type>(0.0));
		return (diffx * diffx + diffy * diffy + diffz * diffz) < (r * r);
	}

	bool OCTreeManager::search_recursive(oc_tree_node_ptr const& root, oc_tree_point_type const& query,
	                                     ResultManagerBase& result_manager) {
		if (root == nullptr) {
			return false;
		}
		#if 0
		for (int i = 0; i < 3; ++i) {
			printf("%f ", root->center()[i]);
		}
		printf("\n");
		if(std::abs(root->center()[0] - (-11.642219)) < 0.000001){
			printf("fuck\n");
		}
		#endif
		if (root->is_leaf()) {
			// 将query点更新进manager
			for (auto idx = 0; idx < root->indices_size(); idx++) {
				auto distance = query.distance_with((*points_)[root->get_indices(idx)]);
				result_manager.update_result(root->get_indices(idx), distance);
			}
			return warst_inside_oct(root, query, result_manager.get_max_distance());
		}
		// 非叶子节点，先到点所在的地方搜索
		uint8_t oct_code = (query[0] > root->center()[0])
		                   | ((query[1] > root->center()[1]) << 1u)
		                   | ((query[2] > root->center()[2]) << 2u);
		if (search_recursive(root->child(oct_code), query, result_manager)) {
			return true;
		}

		for (int i = 0; i < 8; ++i) {
			if (i == oct_code || root->child(i) == nullptr) {
				continue;
			}
			// 不想交就不用检测此节点
			if (!warst_overlap_oct(root->child(i), query, result_manager.get_max_distance())) {
				continue;
			}
			if (search_recursive(root->child(i), query, result_manager)) {
				return true;
			}
		}
		return warst_inside_oct(root, query, result_manager.get_max_distance());
	}

	RadisusResultManager
	OCTreeManager::perform_radius_search_fast(oc_tree_point_type const& query, scalar_type radius) {
		return RadisusResultManager(0);
	}


}