//
// Created by libaoyu on 2020/11/3.
//

#pragma once
#ifndef NNTREES_COMMON_TYPES_H
#define NNTREES_COMMON_TYPES_H

#include <sys/types.h>
#include <vector>
#include <memory>
#include <map>
#include <cassert>
#include <cmath>
#include <sys/time.h>

namespace nn_trees {
	using index_type = int32_t;
	using indices_type = std::vector<index_type>;
	using bool_type = bool;
	using size_type = std::size_t;
	using double_type = double;

	template<typename Scalar, size_type dimension>
	class Point {
		Scalar                 data_[dimension];
		const static size_type dim = dimension;
	public:
		using ScalarType = Scalar;
		using pointer_type = std::shared_ptr<Point<Scalar, dimension>>;
		using vectors_type = std::vector<Point<Scalar, dimension>>;

		Scalar& operator [](size_type idx) {
			assert(idx < dim && "Error bad idx");
			return data_[idx];
		}

		const Scalar& operator [](size_type idx) const {
			assert(idx < dim && "Error bad idx");
			return data_[idx];
		}

		static constexpr size_type size() { return dim; }
		Scalar distance_with(const Point &another)const{
			Scalar sum = 0;
			for (int i = 0; i < dim; ++i) {
				auto tmp = (data_[i] - another.data_[i]);
				sum += tmp*tmp;
			}
			return sum;
		}
	};

	using scalar_type = double;
	using point_3d = Point<scalar_type, 3>;
	using points_3d = Point<scalar_type, 3>::vectors_type;


    class ResultManager {
        // 维护为大顶堆
    public:
        using distance_id_map = std::multimap<scalar_type, index_type, std::greater<>>;
        explicit ResultManager(size_type capacity) : capacity_(capacity) {}

        void update_result(index_type index, scalar_type distance) {
            if (distance < get_max_distance()) {
                // 有必要再插入和删除
//				result_map_[distance] = index;
                result_map_.insert(std::make_pair(distance, index));
                if (result_map_.size() > capacity_) {
                    // 结果集满了，先插入元素，再删除最大元素，最大元素在堆顶
                    result_map_.erase(result_map_.begin());
                }
            }
        }

        scalar_type get_max_distance() {
            if (result_map_.size() < capacity_) {
                return std::numeric_limits<scalar_type>::max();
            }
            return result_map_.begin()->first;
        }

        size_type result_size() {
            return result_map_.size();
        }

        void show_results() {
            for (auto& iter : result_map_) {
                printf("distance: %f idx: %d\n", iter.first, iter.second);
            }
            printf("total %lu result\n", result_map_.size());
        }

        distance_id_map &get_result(){
            return result_map_;
        }
    private:
        size_type       capacity_;
        distance_id_map result_map_;
    };

    class Tictoc{
        struct timeval start{};
        struct timeval stop{};
        size_type last_cnt = 0;
    public:
        void tic(){
            gettimeofday(&start, nullptr);
        }
        size_type toc(){
            gettimeofday(&stop, nullptr);
            last_cnt =  double(stop.tv_sec - start.tv_sec)*1000000.0 + double(stop.tv_usec - start.tv_usec);
            return last_cnt;
        }
        size_type get_last_cnt_us(){return last_cnt;}
        size_type get_last_cnt_ms(){return last_cnt/1000.0;}
    };
}

#endif //NNTREES_COMMON_TYPES_H
