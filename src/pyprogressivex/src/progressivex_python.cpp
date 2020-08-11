#include "progressivex_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "progx_utils.h"
#include "utils.h"
#include "GCoptimization.h"
#include "grid_neighborhood_graph.h"
#include "flann_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "progressive_x.h"

#include <ctime>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>
#include <glog/logging.h>

int find6DPoses_(
	const std::vector<double>& imagePoints,
	const std::vector<double>& worldPoints,
	const std::vector<double>& intrinsicParams,
	std::vector<size_t>& labeling,
	std::vector<double>& poses,
	const double &spatial_coherence_weight,
	const double &threshold,
	const double &confidence,
	const double &neighborhood_ball_radius,
	const double &maximum_tanimoto_similarity,
	const size_t &max_iters,
	const size_t &minimum_point_number,
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	// Calculate the inverse of the intrinsic camera parameters
	Eigen::Matrix3d K;
	K << intrinsicParams[0], intrinsicParams[1], intrinsicParams[2],
		intrinsicParams[3], intrinsicParams[4], intrinsicParams[5],
		intrinsicParams[6], intrinsicParams[7], intrinsicParams[8];
	const Eigen::Matrix3d Kinv =
		K.inverse();
	
	Eigen::Vector3d vec;
	vec(2) = 1;
	size_t num_tents = imagePoints.size() / 2;
	cv::Mat points(num_tents, 5, CV_64F);
	cv::Mat normalized_points(num_tents, 5, CV_64F);
	size_t iterations = 0;
	for (size_t i = 0; i < num_tents; ++i) {
		vec(0) = imagePoints[2 * i];
		vec(1) = imagePoints[2 * i + 1];
		
		points.at<double>(i, 0) = imagePoints[2 * i];
		points.at<double>(i, 1) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
		
		normalized_points.at<double>(i, 0) = Kinv.row(0) * vec;
		normalized_points.at<double>(i, 1) = Kinv.row(1) * vec;
		normalized_points.at<double>(i, 2) = worldPoints[3 * i];
		normalized_points.at<double>(i, 3) = worldPoints[3 * i + 1];
		normalized_points.at<double>(i, 4) = worldPoints[3 * i + 2];
	}
	
	// Normalize the threshold
	const double f = 0.5 * (K(0,0) + K(1,1));
	const double normalized_threshold =
		threshold / f;
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultPnPEstimator, // The type of the used model estimator
		gcransac::sampler::UniformSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	poses.reserve(12 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		poses.emplace_back(model.descriptor(0, 0));
		poses.emplace_back(model.descriptor(0, 1));
		poses.emplace_back(model.descriptor(0, 2));
		poses.emplace_back(model.descriptor(0, 3));
		poses.emplace_back(model.descriptor(1, 0));
		poses.emplace_back(model.descriptor(1, 1));
		poses.emplace_back(model.descriptor(1, 2));
		poses.emplace_back(model.descriptor(1, 3));
		poses.emplace_back(model.descriptor(2, 0));
		poses.emplace_back(model.descriptor(2, 1));
		poses.emplace_back(model.descriptor(2, 2));
		poses.emplace_back(model.descriptor(2, 3));
	}
	
	return progressive_x.getModelNumber();
}

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
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = sourcePoints.size() / 2;
	
	double max_x = std::numeric_limits<double>::min(),
		min_x =  std::numeric_limits<double>::max(),
		max_y = std::numeric_limits<double>::min(),
		min_y =  std::numeric_limits<double>::max();
		
	cv::Mat points(num_tents, 4, CV_64F);
	for (size_t i = 0; i < num_tents; ++i) {
		
		const double 
			&x1 = sourcePoints[2 * i],
			&y1 = sourcePoints[2 * i + 1],
			&x2 = destinationPoints[2 * i],
			&y2 = destinationPoints[2 * i + 1];
		
		max_x = MAX(max_x, x1);
		min_x = MIN(min_x, x1);
		max_x = MAX(max_x, x2);
		min_x = MIN(min_x, x2);
		
		max_y = MAX(max_y, y1);
		min_y = MIN(min_y, y1);
		max_y = MAX(max_y, y2);
		min_y = MIN(min_y, y2);
		
		points.at<double>(i, 0) = x1;
		points.at<double>(i, 1) = y1;
		points.at<double>(i, 2) = x2;
		points.at<double>(i, 3) = y2;
	}
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::ProgressiveNapsacSampler main_sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::DefaultHomographyEstimator::sampleSize(), // The size of a minimal sample
		max_x + std::numeric_limits<double>::epsilon(), // The width of the source image
		max_y + std::numeric_limits<double>::epsilon(), // The height of the source image
		max_x + std::numeric_limits<double>::epsilon(), // The width of the destination image
		max_y + std::numeric_limits<double>::epsilon()); // The height of the destination image

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultHomographyEstimator, // The type of the used model estimator
		gcransac::sampler::ProgressiveNapsacSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}

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
	const int &maximum_model_number)
{
	// Initialize Google's logging library.
	static bool isLoggingInitialized = false;
	if (!isLoggingInitialized)
	{
		google::InitGoogleLogging("pyprogessivex");
		isLoggingInitialized = true;
	}
	
	const size_t num_tents = sourcePoints.size() / 2;
	
	double max_x = std::numeric_limits<double>::min(),
		min_x =  std::numeric_limits<double>::max(),
		max_y = std::numeric_limits<double>::min(),
		min_y =  std::numeric_limits<double>::max();
		
	cv::Mat points(num_tents, 4, CV_64F);
	for (size_t i = 0; i < num_tents; ++i) {
		
		const double 
			&x1 = sourcePoints[2 * i],
			&y1 = sourcePoints[2 * i + 1],
			&x2 = destinationPoints[2 * i],
			&y2 = destinationPoints[2 * i + 1];
		
		max_x = MAX(max_x, x1);
		min_x = MIN(min_x, x1);
		max_x = MAX(max_x, x2);
		min_x = MIN(min_x, x2);
		
		max_y = MAX(max_y, y1);
		min_y = MIN(min_y, y1);
		max_y = MAX(max_y, y2);
		min_y = MIN(min_y, y2);
		
		points.at<double>(i, 0) = x1;
		points.at<double>(i, 1) = y1;
		points.at<double>(i, 2) = x2;
		points.at<double>(i, 3) = y2;
	}
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds

	printf("Neighborhood calculation time = %f secs.\n", elapsed_seconds.count());

	// The main sampler is used inside the local optimization
	gcransac::sampler::ProgressiveNapsacSampler main_sampler(&points, // All data points
		{ 16, 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::DefaultFundamentalMatrixEstimator::sampleSize(), // The size of a minimal sample
		max_x + std::numeric_limits<double>::epsilon(), // The width of the source image
		max_y + std::numeric_limits<double>::epsilon(), // The height of the source image
		max_x + std::numeric_limits<double>::epsilon(), // The width of the destination image
		max_y + std::numeric_limits<double>::epsilon()); // The height of the destination image

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::DefaultFundamentalMatrixEstimator, // The type of the used model estimator
		gcransac::sampler::ProgressiveNapsacSampler, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x(nullptr);

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = threshold;
	// The required confidence in the results
	settings.setConfidence(confidence);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight;
	// Setting the maximum iteration number
	settings.proposal_engine_settings.max_iteration_number = max_iters;
	// Setting the maximum model number if needed
	if (maximum_model_number > 0)
		settings.maximum_model_number = maximum_model_number;

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	// The obtained labeling
	labeling = progressive_x.getStatistics().labeling;
	homographies.reserve(9 * progressive_x.getModelNumber());
	
	// Saving the homography parameters
	for (size_t model_idx = 0; model_idx < progressive_x.getModelNumber(); ++model_idx)
	{
		const auto &model = progressive_x.getModels()[model_idx];
		homographies.emplace_back(model.descriptor(0, 0));
		homographies.emplace_back(model.descriptor(0, 1));
		homographies.emplace_back(model.descriptor(0, 2));
		homographies.emplace_back(model.descriptor(1, 0));
		homographies.emplace_back(model.descriptor(1, 1));
		homographies.emplace_back(model.descriptor(1, 2));
		homographies.emplace_back(model.descriptor(2, 0));
		homographies.emplace_back(model.descriptor(2, 1));
		homographies.emplace_back(model.descriptor(2, 2));
	}
	
	return progressive_x.getModelNumber();
}