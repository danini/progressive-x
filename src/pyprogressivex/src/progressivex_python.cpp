#include "progressivex_python.h"
#include <vector>
#include <thread>
#include "utils.h"
#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include "GCRANSAC.h"
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
#include "solver_epnp_lm.h"

#include "progressive_x.h"

#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

using namespace gcransac;

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
	const bool &log)
{	
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
	cv::Mat points(num_tents, 7, CV_64F);
	cv::Mat points_for_neighborhood(points.rows, 5, CV_64F); // The matrix containing the data from which the neighborhoods are calculated
	std::map<std::pair<int, int>, int> pixels; // A helper variable to count how many pixels are occupied
	
	for (size_t i = 0; i < num_tents; ++i) {		
		points.at<double>(i, 5) = imagePoints[2 * i];
		points.at<double>(i, 6) = imagePoints[2 * i + 1];
		points.at<double>(i, 2) = worldPoints[3 * i];
		points.at<double>(i, 3) = worldPoints[3 * i + 1];
		points.at<double>(i, 4) = worldPoints[3 * i + 2];
			
		vec(0) = points.at<double>(i, 5);
		vec(1) = points.at<double>(i, 6);

		points.at<double>(i, 0) = Kinv.row(0) * vec;
		points.at<double>(i, 1) = Kinv.row(1) * vec;
		
		points_for_neighborhood.at<double>(i, 0) = points.at<double>(i, 5);
		points_for_neighborhood.at<double>(i, 1) = points.at<double>(i, 6);
		points_for_neighborhood.at<double>(i, 2) = points.at<double>(i, 2) * scaling_from_millimeters;
		points_for_neighborhood.at<double>(i, 3) = points.at<double>(i, 3) * scaling_from_millimeters;
		points_for_neighborhood.at<double>(i, 4) = points.at<double>(i, 4) * scaling_from_millimeters;

		points_for_neighborhood.at<double>(i, 0) = points.at<double>(i, 5);
		points_for_neighborhood.at<double>(i, 1) = points.at<double>(i, 6);

		const int x = points.at<double>(i, 5),
			y = points.at<double>(i, 6);
		pixels[std::make_pair(x, y)] = 1;	
	}

	// Normalize the threshold
	const double f = 0.5 * (K(0,0) + K(1,1));
	const double normalized_threshold =
		threshold / f;
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points_for_neighborhood, // All data points
		neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.

	typedef estimator::PerspectiveNPointEstimator<estimator::solver::P3PSolver, // The solver used for fitting a model to a minimal sample
		estimator::solver::EPnPLM> // The solver used for fitting a model to a non-minimal sample
		PnPEstimator;

	// Apply Graph-cut RANSAC
	PnPEstimator estimator(minimum_triangle_size);
	gcransac::Pose6D model;

	// Initialize the samplers
	// The main sampler is used inside the local optimization
	gcransac::sampler::UniformSampler main_sampler(&points);
	//gcransac::sampler::UniformSampler main_sampler(&points);

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);
	
	// Checking if the samplers are initialized successfully.
	if (!main_sampler.isInitialized() ||
		!local_optimization_sampler.isInitialized())
	{
		fprintf(stderr, "One of the samplers is not initialized successfully.\n");
		return 0;
	}
	
	// Applying Progressive-X 
	// (i) if the number of models is not specified, 
	// (ii) if the number of models is known and is greater than 1.
	if (maximum_model_number == -1 ||
		maximum_model_number > 1)
	{		
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
		// The inlier-outlier threshold
		settings.minimum_triangle_size = minimum_triangle_size;
		// The required confidence in the results
		settings.setConfidence(confidence);
		// The maximum Tanimoto similarity of the proposal and compound instances
		settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity;
		// The weight of the spatial coherence term
		settings.spatial_coherence_weight = spatial_coherence_weight;
		// Setting the maximum iteration number
		settings.proposal_engine_settings.max_iteration_number = max_iters;
		// Set the number of pixels covered by an object		
		settings.proposal_engine_settings.used_pixels = pixels.size();
		settings.proposal_engine_settings.minimum_pixel_coverage = minimum_coverage;
		// The weight of the spatial coherence term
		settings.proposal_engine_settings.spatial_coherence_weight = spatial_coherence_weight; 
		// The required confidence in the results
		settings.proposal_engine_settings.confidence = confidence; 
		// The maximum number of local optimizations
		settings.proposal_engine_settings.max_local_optimization_number = 20;
		// The maximum number of iterations
		settings.proposal_engine_settings.max_iteration_number = max_iters; 
		// The minimum number of iterations
		settings.proposal_engine_settings.min_iteration_number = 10; 
		// The radius of the neighborhood ball
		settings.proposal_engine_settings.neighborhood_sphere_radius = 8; 
		// If more than this number of models are supposed to be found, turn of PEARL to speed up the procedure
		settings.maximum_model_number_to_optimize = maximum_model_number_for_optimization;

		// Setting the maximum model number if needed
		if (maximum_model_number > 0)
			settings.maximum_model_number = maximum_model_number;

		progressive_x.log(log);

		progressive_x.run(points, // All data points
			neighborhood, // The neighborhood graph
			estimator, // The used model estimator
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

			scores.emplace_back(progressive_x.getStatistics().scores[model_idx]);
		}
		
		return progressive_x.getModelNumber();
	} else
	{		
		GCRANSAC<PnPEstimator, neighborhood::FlannNeighborhoodGraph, EPOSScoringFunction<PnPEstimator>> gcransac;
		gcransac.setFPS(-1); // Set the desired FPS (-1 means no limit)
		gcransac.settings.threshold = normalized_threshold; // The inlier-outlier threshold
		gcransac.settings.minimum_pixel_coverage = minimum_coverage;
		gcransac.settings.spatial_coherence_weight = spatial_coherence_weight; // The weight of the spatial coherence term
		gcransac.settings.confidence = proposal_engine_confidence; // The required confidence in the results
		gcransac.settings.max_local_optimization_number = 20; // The maximum number of local optimizations
		gcransac.settings.max_iteration_number = max_iters; // The maximum number of iterations
		gcransac.settings.min_iteration_number = 10; // The minimum number of iterations
		gcransac.settings.neighborhood_sphere_radius = 8; // The radius of the neighborhood ball
		gcransac.settings.core_number = 1; // The number of parallel processes
		gcransac.settings.used_pixels = pixels.size();

		// Start GC-RANSAC
		gcransac.run(points,
			estimator,
			&main_sampler,
			&local_optimization_sampler,
			&neighborhood,
			model);
		
		const utils::RANSACStatistics &statistics = gcransac.getRansacStatistics();
		const size_t inlier_number = statistics.inliers.size();

		if (log)
		{
			printf("Inlier number = %d\nIteration number = %d\n", inlier_number,
				statistics.iteration_number);
		}

		scores.emplace_back(statistics.score);

		// If numerical optimization is needed, apply the Levenberg-Marquardt 
		// implementation of OpenCV.
		if (apply_numerical_optimization && 
			inlier_number >= 6)
		{
			// The estimated rotation matrix
			Eigen::Matrix3d rotation =
				model.descriptor.leftCols<3>();
			// The estimated translation
			Eigen::Vector3d translation =
				model.descriptor.rightCols<1>();

			// Copy the data into two matrices containing the image and object points. 
			// This would not be necessary, but selecting the submatrices by cv::Rect
			// leads to an error in cv::solvePnP().
			cv::Mat inlier_image_points(inlier_number, 2, CV_64F),
				inlier_object_points(inlier_number, 3, CV_64F);

			for (size_t i = 0; i < inlier_number; ++i)
			{
				const size_t &idx = statistics.inliers[i];
				inlier_image_points.at<double>(i, 0) = points.at<double>(idx, 0);
				inlier_image_points.at<double>(i, 1) = points.at<double>(idx, 1);
				inlier_object_points.at<double>(i, 0) = points.at<double>(idx, 2);
				inlier_object_points.at<double>(i, 1) = points.at<double>(idx, 3);
				inlier_object_points.at<double>(i, 2) = points.at<double>(idx, 4);
			}

			// Converting the estimated pose parameters OpenCV format
			cv::Mat cv_rotation(3, 3, CV_64F, rotation.data()), // The estimated rotation matrix converted to OpenCV format
				cv_translation(3, 1, CV_64F, translation.data()); // The estimated translation converted to OpenCV format
			
			// Convert the rotation matrix by the rodrigues formula
			cv::Mat cv_rodrigues(3, 1, CV_64F);
			cv::Rodrigues(cv_rotation.t(), cv_rodrigues);

			// Applying numerical optimization to the estimated pose parameters
			cv::solvePnP(inlier_object_points, // The object points
				inlier_image_points, // The image points
				cv::Mat::eye(3, 3, CV_64F), // The camera's intrinsic parameters 
				cv::Mat(), // An empty vector since the radial distortion is not known
				cv_rodrigues, // The initial rotation
				cv_translation, // The initial translation
				true, // Use the initial values
				cv::SOLVEPNP_ITERATIVE); // Apply numerical refinement
			
			// Convert the rotation vector back to a rotation matrix
			cv::Rodrigues(cv_rodrigues, cv_rotation);

			// Transpose the rotation matrix back
			//cv_rotation = cv_rotation.t();

			// Calculate the error of the refined pose
			model.descriptor <<
				cv_rotation.at<double>(0, 0), cv_rotation.at<double>(0, 1), cv_rotation.at<double>(0, 2), cv_translation.at<double>(0),
				cv_rotation.at<double>(1, 0), cv_rotation.at<double>(1, 1), cv_rotation.at<double>(1, 2), cv_translation.at<double>(1),
				cv_rotation.at<double>(2, 0), cv_rotation.at<double>(2, 1), cv_rotation.at<double>(2, 2), cv_translation.at<double>(2);
		}
		
		double score = statistics.score;
		labeling.resize(num_tents);

		const int num_inliers = statistics.inliers.size();
		for (auto pt_idx = 0; pt_idx < num_tents; ++pt_idx) {
			labeling[pt_idx] = 0;
		}

		for (auto pt_idx = 0; pt_idx < num_inliers; ++pt_idx) {
			labeling[statistics.inliers[pt_idx]] = 1;
		}

		poses.resize(12);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				poses[i * 4 + j] = (double)model.descriptor(i, j);
			}
		}
	}
	
	return 1;
}