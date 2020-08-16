#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>

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

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>

struct stat info;

enum Problem { Homography, TwoViewMotion, RigidMotion, Pose6D };

void testMulti6DPoseFitting(
	const std::string &scene_name_, // The name of the current scene 
	const std::string &input_correspondence_path_, // The path of the detected correspondences
	const std::string &camera_intrinsics_path_, // The path of the intrinsic camera parameters
	const std::string &ground_truth_path_, // The path of the ground truth poses 
	const std::string &output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const double neighborhood_ball_radius_,
	const double maximum_tanimoto_similarity_,
	const size_t minimum_point_number_);

double rotationError(const Eigen::Matrix3d &reference_rotation_,
	const Eigen::Matrix3d &estimated_rotation_);

void drawMatches(
	const cv::Mat &points_,
	const std::vector<size_t> &inliers_,
	const cv::Mat &image_src_,
	const cv::Mat &image_dst_,
	cv::Mat &out_image_,
	int circle_radius_,
	const cv::Scalar &color_);

std::mutex writing_mutex;
int settings_number = 0;

std::vector<std::string> getAvailableTestScenes(const Problem &problem_);

int main(int argc, const char* argv[])
{
	const std::string root_directory = ""; // The directory where the 'data' folder is found

	const bool visualize_results = true, // A flag to tell if the resulting labeling should be visualized
		visualize_inner_steps = false; // A flag to tell if the steps of the algorithm should be visualized
	
	const double confidence = 0.9, // The required confidence in the results
		maximum_tanimoto_similarity = 0.9, // The maximum tanimoto similarity used to reject models early
		spatial_coherence_weight = 0.1, // The spatial coherence weight used both in PEARL and GC-RANSAC
		neighborhood_ball_radius = 20.0; // The radius of the ball hyper-sphere used for determining the neighborhood graph.

	for (const std::string &scene : getAvailableTestScenes(Problem::Pose6D))
	{
		const size_t minimum_point_number = 2 * 3; // The minimum number of inliers needed to accept a model instance, i.e., two times the sample size
		const double inlier_outlier_threshold = 4.0; // The inlier-outlier threshold used to assign points to models

		printf("Processed scene = %s.\n", scene.c_str());

		const std::string input_correspondence_path =
			root_directory + "data/" + scene + "/" + scene + ".txt"; // Path where the detected correspondences are saved
		const std::string intrinsics_path =
			root_directory + "data/" + scene + "/" + scene + "_intrinsics.txt"; // Path where the intrinsic camera parameters are
		const std::string ground_truth_path =
			root_directory + "data/" + scene + "/" + scene + "_poses.txt"; // Path where the ground truth poses are
		const std::string  output_correspondence_path =
			root_directory + "results/" + scene + "/result_" + scene + ".txt"; // Path where the inlier correspondences are saved

		testMulti6DPoseFitting(
			scene, // The name of the current scene
			input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
			intrinsics_path,// Path where the intrinsic camera parameters are
			ground_truth_path, // Path where the ground truth poses are
			output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
			confidence, // The RANSAC confidence value
			inlier_outlier_threshold, // The used inlier-outlier threshold in GC-RANSAC.
			spatial_coherence_weight, // The weight of the spatial coherence term in the graph-cut energy minimization.
			neighborhood_ball_radius, // The radius of the neighborhood ball for determining the neighborhoods.
			maximum_tanimoto_similarity, // The maximum Tanimoto similarity of the proposal and compound instances.
			minimum_point_number); // The minimum number of inlier for a model to be kept.
	}

	return 0;
}

std::vector<std::string> getAvailableTestScenes(const Problem &problem_)
{
	switch (problem_)
	{
	case Problem::Pose6D:
		// A scene from the T-LESS dataset. Correspondences are obtained by the EPOS method.
		return { "tless" }; 
	default:
		return {};
	}
}

void testMulti6DPoseFitting(
	const std::string &scene_name_, // The name of the current scene 
	const std::string &input_correspondence_path_, // The path of the detected correspondences
	const std::string &camera_intrinsics_path_, // The path of the intrinsic camera parameters
	const std::string &ground_truth_path_, // The path of the ground truth poses 
	const std::string &output_correspondence_path_,  // The path of the correspondences saved with their labels
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double spatial_coherence_weight_, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double neighborhood_ball_radius_, // The radius of the neighborhood ball for determining the neighborhoods.
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const size_t minimum_point_number_) // The minimum number of inlier for a model to be kept.
{
	// Loading the 2D-3D correspondences from file
	cv::Mat points;
	gcransac::utils::loadPointsFromFile<5>(points, input_correspondence_path_.c_str());
	
	// Load the camera matrices
	Eigen::Matrix3d intrinsics;
	gcransac::utils::loadMatrix<double, 3, 3>(camera_intrinsics_path_, intrinsics);

	// Load the ground truth poses
	cv::Mat ground_truth_poses;
	gcransac::utils::loadPointsFromFile<12>(ground_truth_poses, ground_truth_path_.c_str());

	// Normalize the correspondences by the intrinsic camera matrices
	cv::Mat normalized_image_points(points.rows, 2, CV_64F);
	gcransac::utils::normalizeImagePoints(
		points(cv::Rect(0, 0, 2, points.rows)),
		intrinsics,
		normalized_image_points);

	cv::Mat normalized_points(points.rows, 7, CV_64F);
	
	// Copy the normalized 2D points to the last columns. 
	normalized_image_points.copyTo(normalized_points(cv::Rect(0, 0, 2, points.rows)));
	// Copy the original 2D points to the last columns. They will be used for degeneracy testing
	points(cv::Rect(0, 0, 2, points.rows)).copyTo(normalized_points(cv::Rect(5, 0, 2, points.rows)));
	// Copy the 3D points to the matrix
	points(cv::Rect(2, 0, 3, points.rows)).copyTo(normalized_points(cv::Rect(2, 0, 3, points.rows)));

	// Normalize the threshold
	const double f = 
		(intrinsics(0, 0) + intrinsics(1, 1)) / 2.0;
	const double normalized_threshold =
		inlier_outlier_threshold_ / f;

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		neighborhood_ball_radius_); // The radius of the neighborhood ball for determining the neighborhoods.
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
	settings.minimum_number_of_inliers = minimum_point_number_;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = normalized_threshold;
	// The required confidence in the results
	settings.setConfidence(confidence_);
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity_;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = spatial_coherence_weight_;
	// The maximum iteration number of GC-RANSAC
	settings.proposal_engine_settings.max_iteration_number = 1000;

	gcransac::utils::DefaultPnPEstimator estimator;

	progressive_x.run(normalized_points, // All data points
		neighborhood, // The neighborhood graph
		estimator, // The used estimator
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	
	printf("Processing time = %f secs.\n", progressive_x.getStatistics().processing_time);
	printf("Number of found model instances = %d (there are %d instances in the reference labeling).\n", progressive_x.getModelNumber(), ground_truth_poses.rows);

	for (size_t pose_idx = 0; pose_idx < ground_truth_poses.rows; ++pose_idx)
	{
		const Eigen::Map < Eigen::Matrix<double, 3, 4, Eigen::RowMajor> >
			pose(ground_truth_poses.row(pose_idx).ptr<double>());
		const Eigen::Vector3d &gt_translation = pose.leftCols<1>();
		const Eigen::Matrix3d &gt_rotation = pose.block<3, 3>(0, 0);

		double best_rotation_error = std::numeric_limits<double>::max() / 2.0,
			best_translation_error = std::numeric_limits<double>::max() / 2.0;

		for (const auto &model : progressive_x.getModels())
		{
			const Eigen::Vector3d &translation = model.descriptor.leftCols<1>();
			const Eigen::Matrix3d &rotation = model.descriptor.block<3, 3>(0, 0);

			double rotation_error,
				translation_error;

			translation_error = (gt_translation - translation).norm();
			rotation_error = rotationError(gt_rotation, rotation);

			if (translation_error + rotation_error < best_rotation_error + best_translation_error)
			{
				best_translation_error = translation_error;
				best_rotation_error = rotation_error;
			}
		}

		printf("%d-th pose's error\n\tRotation error = %f\370\n\tTranslation error = %f mm\n",
			pose_idx + 1,
			best_rotation_error,
			best_translation_error);
	}
}

double rotationError(const Eigen::Matrix3d &reference_rotation_,
	const Eigen::Matrix3d &estimated_rotation_)
{
	constexpr double radian_to_degree_multiplier = 180.0 / M_PI;

	const double trace_R_est_times_R_ref =
		(estimated_rotation_ * reference_rotation_.transpose()).trace();

	double error_cos = 0.5 * (trace_R_est_times_R_ref - 1.0);

	// Avoid invalid values due to numerical errors.
	error_cos = std::clamp(error_cos, -1.0, 1.0);

	return radian_to_degree_multiplier * std::acos(error_cos);
}