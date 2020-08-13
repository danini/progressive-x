#include <vector>
#include <string>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& homographies,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const double &scaling_from_millimeters,
	const double &minimum_coverage,
	const double &minimum_triangle_size,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number);

int findHomographies_(
		const std::vector<double>& sourcePoints,
		const std::vector<double>& destinationPoints,
		std::vector<size_t>& labeling,
		std::vector<double>& homographies,
		const double &spatial_coherence_weight,
		const double &threshold,
		const double &confidence,
		const double &neighborhood_ball_radius,
		const double &maximum_tanimoto_similarity,
		const size_t &max_iters,
		const size_t &minimum_point_number,
		const int &maximum_model_number);
		
int findTwoViewMotions_(
		const std::vector<double>& sourcePoints,
		const std::vector<double>& destinationPoints,
		std::vector<size_t>& labeling,
		std::vector<double>& homographies,
		const double &spatial_coherence_weight,
		const double &threshold,
		const double &confidence,
		const double &neighborhood_ball_radius,
		const double &maximum_tanimoto_similarity,
		const size_t &max_iters,
		const size_t &minimum_point_number,
		const int &maximum_model_number);