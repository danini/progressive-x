#include <vector>
#include <string>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	std::vector<double>& scores,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &proposal_engine_confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const double &scaling_from_millimeters,
	const double &minimum_coverage,
	const double &minimum_triangle_size,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number,
	const bool &use_prosac,
	const size_t &maximum_model_number_for_optimization,
	const bool &apply_numerical_optimization,
	const bool &log);
