//
// Created by libaoyu on 2020/11/3.
//

#ifndef NNTREES_KDTREE_H
#define NNTREES_KDTREE_H

#include <memory>
#include <utility>
#include "common_types.h"

namespace nn_trees {

    template<class PointType>
    class KDTreeNode {
    public:
        using node_pointer_type = std::shared_ptr<KDTreeNode>;
        using indices_type = std::vector<index_type>;
        using point_type = PointType;
        using point_ptr_type = std::shared_ptr<point_type>;

        const static size_type K = PointType::size();

        KDTreeNode(
                size_type axis,
                scalar_type value,
                node_pointer_type left,
                node_pointer_type right,
                index_type idx_start, index_type index_end
        ) : axis_(axis),
            value_(value),
            left_(std::move(left)),
            right_(std::move(right)), start_(idx_start), end_(idx_end()) {
            assert(axis_ < K);
        }

        static node_pointer_type create_node(
                size_type axis,
                scalar_type value,
                const node_pointer_type &left,
                const node_pointer_type &right,
                index_type idx_start, index_type idx_end
        ) {
            return std::make_shared<KDTreeNode>(axis, value, left, right, idx_start, idx_end);
        }

        bool_type is_leaf() const {
            return left_ == nullptr && right_ == nullptr;
        }

        node_pointer_type &left() { return left_; }

        node_pointer_type &right() { return right_; }

        scalar_type &value() { return value_; }

        scalar_type &split_value() {
            return value_;
        }

        [[nodiscard]] const node_pointer_type &left() const { return left_; }

        [[nodiscard]] const node_pointer_type &right() const { return right_; }

        [[nodiscard]] size_type get_axis() const { return axis_; }

        index_type idx_start() const { return start_; }

        index_type idx_end() const { return end_; }

    private:
        size_type axis_;
        scalar_type value_;
        node_pointer_type left_;
        node_pointer_type right_;
        // 避免内存占用，只用start和end来表示下标
        index_type start_;
        index_type end_;
    };

    using kdtree_3d_node_type = KDTreeNode<point_3d>;
    using kdtree_3d_node_ptr = kdtree_3d_node_type::node_pointer_type;

    kdtree_3d_node_ptr build_tree(const points_3d &points, size_type leaf_size);

    ResultManager
    knn_search_kdtree(const kdtree_3d_node_ptr &root,
                      const points_3d &all_points, const point_3d &query_point,
                      int result_size);

    ResultManager knn_search_bf(const points_3d &all_points, const point_3d &query_point, int result_size);

    void kdtree_traverse(kdtree_3d_node_ptr &root, indices_type &test_data);
}


#endif //NNTREES_KDTREE_H

