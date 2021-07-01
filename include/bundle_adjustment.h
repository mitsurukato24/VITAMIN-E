#pragma once
#include "common_types.h"


class SubspaceGaussNewton
{
public:
    SubspaceGaussNewton(Camera::Ptr cam) {}

private:
    Camera::Ptr cam_;
    
    // Each frame has pose
    using FrameIdPair = std::pair<unsigned int, unsigned int>;
    std::unordered_map<FrameIdPair, Eigen::Matrix<float, 6, 6>, pair_hash<FrameIdPair>, std::equal_to<FrameIdPair>, Eigen::aligned_allocator<std::pair<const FrameIdPair, MatchData>>> match_data_;
};


class JacobianAccumulator
{
public:
    JacobianAccumulator() {}

};
