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
#include "progress_visualizer.h"

#include <ctime>
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
	#include <direct.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>

#include <mutex>
#include <gflags/gflags.h>
#include <glog/logging.h>

/*
	Initializing the flags
*/
DEFINE_string(data_path, "data/",
	"The folder where the images are stored.");
DEFINE_string(statistics_path, "lineSegmentResults.csv",
	"The folder where the results are saved in .csv format.");
DEFINE_int32(number_of_test_images, 321,
	"The number of test scenes in the folder.");
DEFINE_bool(draw_results, false,
	"A flag determining if the results should be drawn and shown.");
DEFINE_double(confidence, 0.9999,
	"The confidence of the multi-model fitting.");
DEFINE_double(threshold, 1.5,
	"The inlier-outlier threshold.");
DEFINE_double(neighborhood_ball_radius, 20.0,
	"The neighborhood ball's radius.");	
DEFINE_int32(maximum_iterations, 100,
	"The maximum round of multi-model fitting.");
DEFINE_double(spatial_coherence_weight, 0.1,
	"The weight used in the spatial coherence energy.");
DEFINE_double(maximum_tanimoto_similarity, 0.5,
	"The maximum accepted similarity in the clustering. A values in-between [0, 1].");
DEFINE_int32(minimum_point_number, 3,
	"The minimum number of points required to keep a model.");
DEFINE_int32(core_number, 1,
	"The number of cores used for processing the dataset.");
DEFINE_int32(repetitions, 5,
	"The number of repetitions.");
	
void multiLineSegmentFitting(
	const std::string &data_path_,
	const std::string &ground_truth_path_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_); // A flag to determine if the results should be visualized

typedef progx::Model<gcransac::utils::Default2DLineEstimator> ModelType;

void recoverLineSegments(
	const cv::Mat &points_,
	const std::vector<ModelType> &lines_,
	const std::vector<std::vector<size_t>> &lineInliers_,
	std::vector<Eigen::Vector4d> &lineSegments_);

double calculateError(
	const cv::Mat &groundTruthPolygon_,
	const std::vector<Eigen::Vector4d> &lineSegments_);

int main(int argc, char** argv)
{
	// Parsing the flags
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Initialize Google's logging library.
	google::InitGoogleLogging(argv[0]);

	if (FLAGS_core_number > 1 && FLAGS_draw_results)
	{
		fprintf(stderr, "We do not recommend to set the --draw-results flag true while the --core-number is not 1\n");
		return 0;
	}

	LOG(INFO) << "The used parameters are:" <<
		 "\n\tNumber of cores = " << FLAGS_core_number <<
		 "\n\tInlier-outlier threshold = " << FLAGS_threshold <<
		 "\n\tMinimum point number = " << FLAGS_minimum_point_number <<
		 "\n\tConfidence = " << FLAGS_confidence <<
		 "\n\tMaximum Tanimoto similarity = " << FLAGS_maximum_tanimoto_similarity;

#pragma omp parallel for num_threads(FLAGS_core_number)
	for (int imageIdx = 1; imageIdx <= FLAGS_number_of_test_images; ++imageIdx)
	{
		const std::string data_path =
			FLAGS_data_path + "point2d_" + std::to_string(imageIdx) + ".txt";
		const std::string ground_truth_path =
			FLAGS_data_path + "poly_" + std::to_string(imageIdx) + ".txt";

		for (size_t repetition = 0; repetition < FLAGS_repetitions; ++repetition)
			multiLineSegmentFitting(
				data_path,
				ground_truth_path,
				FLAGS_confidence,
				FLAGS_threshold,
				FLAGS_maximum_iterations,
				FLAGS_maximum_tanimoto_similarity,
				FLAGS_minimum_point_number,
				FLAGS_draw_results);
	}
	
	return 0;
}

void multiLineSegmentFitting(
	const std::string &data_path_,
	const std::string &ground_truth_path_,
	const double confidence_, // The RANSAC confidence value
	const double inlier_outlier_threshold_, // The used inlier-outlier threshold in GC-RANSAC.
	const double maximum_iterations, // The weight of the spatial coherence term in the graph-cut energy minimization.
	const double maximum_tanimoto_similarity_, // The maximum Tanimoto similarity of the proposal and compound instances.
	const double minimum_point_number_, // The minimum number of inlier for a model to be kept.
	const bool visualize_results_) // A flag to determine if the results should be visualized
{
	static std::mutex savingMutex;

	// Load the point coordinates
	cv::Mat points;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		points,
		data_path_.c_str());

	// Loading the ground thruth polygon's coordinates
	cv::Mat groundTruthPolygon;
	gcransac::utils::loadPointsFromFile<2, 1, false>(
		groundTruthPolygon,
		ground_truth_path_.c_str());

	LOG(INFO) << points.rows << " points are loaded.";

	// Calculate bounding box for the points
	Eigen::Vector4d boundingBox(std::numeric_limits<double>::max(),
		std::numeric_limits<double>::max(), 
		std::numeric_limits<double>::lowest(), 
		std::numeric_limits<double>::lowest());

	for (size_t pointIdx = 0; pointIdx < points.rows; ++pointIdx)
	{
		boundingBox(0) = MIN(boundingBox(0), points.at<double>(pointIdx, 0));
		boundingBox(1) = MIN(boundingBox(1), points.at<double>(pointIdx, 1));
		boundingBox(2) = MAX(boundingBox(2), points.at<double>(pointIdx, 0));
		boundingBox(3) = MAX(boundingBox(3), points.at<double>(pointIdx, 1));	
	}

	double totalTime = 0.0,
		time = 0.0;

	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&points, // All data points
		FLAGS_neighborhood_ball_radius); // The radius of the neighborhood ball for determining the neighborhoods.
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	time = elapsed_seconds.count();
	totalTime += time;

	LOG(INFO) << "Neighborhood calculation time = " << time << " secs.";

	// The main sampler is used inside the local optimization
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	gcransac::sampler::ProgressiveNapsacSampler<2> main_sampler(&points, // All data points
		{ 8, 4, 2 }, // The layer structure of the sampler's multiple grids
		gcransac::utils::Default2DLineEstimator::sampleSize(), // The size of a minimal sample
		{ boundingBox(2) - boundingBox(0), // The width of the scene
			boundingBox(3) - boundingBox(2) }); // The height of the destination image
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	elapsed_seconds = end - start; // The elapsed time in seconds
	time = elapsed_seconds.count();
	totalTime += time;

	LOG(INFO) << "Sampler initialization time = " << time << " secs.";

	// The local optimization sampler is used inside the local optimization
	gcransac::sampler::UniformSampler local_optimization_sampler(&points);

	// Applying Progressive-X
	progx::ProgressiveX<gcransac::neighborhood::FlannNeighborhoodGraph, // The type of the used neighborhood-graph
		gcransac::utils::Default2DLineEstimator, // The type of the used model estimator
		gcransac::sampler::ProgressiveNapsacSampler<2>, // The type of the used main sampler in GC-RANSAC
		gcransac::sampler::UniformSampler> // The type of the used sampler in the local optimization of GC-RANSAC
		progressive_x;

	// Set the parameters of Progressive-X
	progx::MultiModelSettings &settings = progressive_x.getMutableSettings();
	// The minimum number of inlier required to keep a model instance.
	// This value is used to determine the label cost weight in the alpha-expansion of PEARL.
	settings.minimum_number_of_inliers = minimum_point_number_;
	// The inlier-outlier threshold
	settings.inlier_outlier_threshold = inlier_outlier_threshold_;
	// The required confidence in the results
	settings.confidence = confidence_;
	// The maximum Tanimoto similarity of the proposal and compound instances
	settings.maximum_tanimoto_similarity = maximum_tanimoto_similarity_;
	// The weight of the spatial coherence term
	settings.spatial_coherence_weight = FLAGS_spatial_coherence_weight;	

	progressive_x.run(points, // All data points
		neighborhood, // The neighborhood graph
		main_sampler, // The main sampler used in GC-RANSAC
		local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
	totalTime += progressive_x.getStatistics().processing_time;

	LOG(INFO) << "Progressive-X processing time = " << progressive_x.getStatistics().processing_time << " secs";
	LOG(INFO) << "Number of found instances = " << progressive_x.getModelNumber();

	const std::vector<ModelType> &models = 
		progressive_x.getModels();
	
	const std::vector<std::vector<size_t>> &modelInliers = 
		progressive_x.getStatistics().inliers_of_each_model;

	// Recover line segments from the obtained 2D lines.
	std::vector<Eigen::Vector4d> lineSegments;
	recoverLineSegments(
		points,
		models,
		modelInliers,
		lineSegments);

	// Calculate the error of the line segments
	if (lineSegments.size() > 0)
	{
		double error = 
			calculateError(groundTruthPolygon,
				lineSegments);

		LOG(INFO) << "The error is " << error << " px.";
		LOG(INFO) << "The total run-time is " << totalTime << " secs.";
		
		savingMutex.lock();
		std::ofstream file(FLAGS_statistics_path, std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			confidence_ << ";" <<
			error << ";" <<
			totalTime << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	} else
	{		
		LOG(INFO) << "No models are found.";

		savingMutex.lock();
		std::ofstream file(FLAGS_statistics_path, std::fstream::app);
		file <<
			inlier_outlier_threshold_ << ";" <<
			maximum_tanimoto_similarity_ << ";" <<
			maximum_iterations << ";" <<
			minimum_point_number_ << ";" <<
			confidence_ << ";" <<
			std::numeric_limits<double>::max() << ";" <<
			totalTime << ";" <<
			models.size() << "\n";
		savingMutex.unlock();
	}

}

double calculateError(
	const cv::Mat &groundTruthPolygon_,
	const std::vector<Eigen::Vector4d> &lineSegments_)
{
	// Calculate the error from the ground truth
	Eigen::VectorXd distances(groundTruthPolygon_.rows);
	for (size_t pointIdx = 0; pointIdx < groundTruthPolygon_.rows; ++pointIdx)
		distances(pointIdx) = std::numeric_limits<double>::max();

	for (size_t modelIdx = 0; modelIdx < lineSegments_.size(); ++modelIdx)
	{
		const auto& lineSegment = lineSegments_[modelIdx];

		Eigen::Vector3d v1, v2;
		v1 << lineSegment.head<2>(), 1;
		v2 << lineSegment.tail<2>(), 1;

		double d = 0;

		for (size_t segmentIdx = 0; segmentIdx < groundTruthPolygon_.rows; ++segmentIdx)
		{
			Eigen::Vector3d gtSegment;
			gtSegment(0) = groundTruthPolygon_.at<double>(segmentIdx, 0);
			gtSegment(1) = groundTruthPolygon_.at<double>(segmentIdx, 1);
			gtSegment(2) = 1;

			Eigen::Vector3d
				a = v1 - v2,
				b = gtSegment - v2,
				c = gtSegment - v1;

			double D1 = abs(a.cross(b)(2)) / sqrt(a(0) * a(0) + a(1) * a(1));
			double D2 = sqrt(c(0) * c(0) + c(1) * c(1));
			double D3 = sqrt(b(0) * b(0) + b(1) * b(1));

			bool insegment = a.dot(b) * (-a).dot(c) >= 0;
			if (insegment)
				d = D1;
			else
				d = MIN(D2, D3);

			distances(segmentIdx) = 
				MIN(distances(segmentIdx), d);
		}
	}

	return distances.mean();
}

void recoverLineSegments(
	const cv::Mat &points_,
	const std::vector<ModelType> &lines_,
	const std::vector<std::vector<size_t>> &lineInliers_,
	std::vector<Eigen::Vector4d> &lineSegments_)
{
	// Occupy the memory for the lines
	lineSegments_.reserve(lines_.size());
	
	// Calculate the end points of the line segments
	for (size_t lineIdx = 0; lineIdx < lineInliers_.size(); ++lineIdx)
	{
		// The line's parameters in their implicit form
		const double
			& a = lines_[lineIdx].descriptor(0),
			& b = lines_[lineIdx].descriptor(1),
			& c = lines_[lineIdx].descriptor(2);

		// The tangent direction of the line
		Eigen::Vector2d lineTangent;
		lineTangent << b, -a;

		// A point on the line
		Eigen::Vector2d pointOnLine;
		pointOnLine << 0, -c / b;

		const auto& inliers = lineInliers_[lineIdx];

		if (inliers.size() < FLAGS_minimum_point_number)
			continue;

		lineSegments_.resize(lineSegments_.size() + 1);
		auto& lineSegment = lineSegments_.back();
		double minParameter = std::numeric_limits<double>::max(),
			maxParameter = std::numeric_limits<double>::lowest(),
			parameter;
		Eigen::Vector2d point;

		// The coefficient matrix to the system projecting each 2D points to the line
		Eigen::Matrix3d coefficients;
		coefficients << 2, 0, a,
			0, 2, b,
			a, b, 0;
		const Eigen::Matrix3d &coefficientsTransposed =
			coefficients;
		const Eigen::Matrix3d covariance =
			coefficientsTransposed * coefficients;
		Eigen::Vector3d inhomogeneousPart;
		inhomogeneousPart(2) = -c;

		// Projecting each inlier to the line
		for (const auto& inlierIdx : inliers)
		{
			inhomogeneousPart(0) = 2 * points_.at<double>(inlierIdx, 0);
			inhomogeneousPart(1) = 2 * points_.at<double>(inlierIdx, 1);

			point = covariance.llt().solve(coefficientsTransposed * inhomogeneousPart).head<2>();
			parameter = (point(0) - pointOnLine(0)) / lineTangent(0);

			// Check if the current parameter is smaller than the smallest so far
			if (parameter < minParameter)
			{
				minParameter = parameter;
				lineSegment(0) = point(1);
				lineSegment(1) = point(0);
			}

			// Check if the current parameter is greater than the greatest so far
			if (parameter > maxParameter)
			{
				maxParameter = parameter;
				lineSegment(2) = point(1);
				lineSegment(3) = point(0);
			}
		}
	}
	
}