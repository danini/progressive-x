#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Eigen>

#include "utils.h"
#include "grid_neighborhood_graph.h"
#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "progressive_x.h"
#include "progress_visualizer.h"

#include <ctime>
#include <direct.h>
#include <sys/types.h>
#include <sys/stat.h>

struct stat info;

void testMultiHomographyFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_);

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_);

void drawMatches(
	const cv::Mat &points_,
	const std::vector<size_t> &inliers_,
	const cv::Mat &image_src_,
	const cv::Mat &image_dst_,
	cv::Mat &out_image_,
	int circle_radius_,
	const cv::Scalar &color_);

using namespace gcransac;

int main(int argc, const char* argv[])
{
	std::string scene = "johnssona";

	printf("Processed scene = '%s'\n", scene.c_str());
	std::string src_image_path, // Path of the source image
		dst_image_path, // Path of the destination image
		input_correspondence_path, // Path where the detected correspondences are saved
		output_correspondence_path, // Path where the inlier correspondences are saved
		output_matched_image_path; // Path where the matched image is saved

	// Initializing the paths
	initializeScene(scene,
		src_image_path,
		dst_image_path,
		input_correspondence_path,
		output_correspondence_path,
		output_matched_image_path);

	testMultiHomographyFitting(
		src_image_path, // The source image's path
		dst_image_path, // The destination image's path
		input_correspondence_path, // The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
		output_correspondence_path, // The path where the inliers of the estimated fundamental matrices will be saved
		output_matched_image_path, // The path where the matched image pair will be saved
		0.99, // The RANSAC confidence value
		1.0, // The used inlier-outlier threshold in GC-RANSAC.
		0.14, // The weight of the spatial coherence term in the graph-cut energy minimization.
		8, // The radius of the neighborhood ball for determining the neighborhoods.
		-1); // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.

	return 0;
}

std::vector<std::string> getAvailableTestScenes()
{
	return { "johnssona" };
}

bool initializeScene(const std::string &scene_name_,
	std::string &src_image_path_,
	std::string &dst_image_path_,
	std::string &input_correspondence_path_,
	std::string &output_correspondence_path_,
	std::string &output_matched_image_path_)
{
	// The root directory where the "results" and "data" folder are
	const std::string root_dir = "";

	// The directory to which the results will be saved
	std::string dir = root_dir + "results/" + scene_name_;

	// Create the task directory if it doesn't exist
	if (stat(dir.c_str(), &info) != 0) // Check if exists
		if (_mkdir(dir.c_str()) != 0) // Create it, if not
		{
			fprintf(stderr, "Error while creating a new folder in 'results'\n");
			return false;
		}

	// The source image's path
	src_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.jpg";
	if (cv::imread(src_image_path_).empty())
		src_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "1.png";
	if (cv::imread(src_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", src_image_path_);
		return false;
	}

	// The destination image's path
	dst_image_path_ =
		root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.jpg";
	if (cv::imread(dst_image_path_).empty())
		dst_image_path_ = root_dir + "data/" + scene_name_ + "/" + scene_name_ + "2.png";
	if (cv::imread(dst_image_path_).empty())
	{
		fprintf(stderr, "Error while loading image '%s'\n", dst_image_path_);
		return false;
	}

	// The path where the detected correspondences (before the robust estimation) will be saved (or loaded from if exists)
	input_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/" + scene_name_ + "_points_with_no_annotation.txt";
	// The path where the inliers of the estimated fundamental matrices will be saved
	output_correspondence_path_ =
		root_dir + "results/" + scene_name_ + "/result_" + scene_name_ + ".txt";
	// The path where the matched image pair will be saved
	output_matched_image_path_ =
		root_dir + "results/" + scene_name_ + "/matches_" + scene_name_ + ".png";

	return true;
}


void testMultiHomographyFitting(
	const std::string &source_path_,
	const std::string &destination_path_,
	const std::string &out_correspondence_path_,
	const std::string &in_correspondence_path_,
	const std::string &output_match_image_path_,
	const double confidence_,
	const double inlier_outlier_threshold_,
	const double spatial_coherence_weight_,
	const size_t cell_number_in_neighborhood_graph_,
	const int fps_)
{
	// Read the images
	cv::Mat source_image = cv::imread(source_path_); // The source image
	cv::Mat destination_image = cv::imread(destination_path_); // The destination image

	if (source_image.empty()) // Check if the source image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			source_path_.c_str());
		return;
	}

	if (destination_image.empty()) // Check if the destination image is loaded successfully
	{
		printf("An error occured while loading image '%s'\n",
			destination_path_.c_str());
		return;
	}

	// Detect or load point correspondences using AKAZE 
	cv::Mat points;
	detectFeatures(
		in_correspondence_path_, // The path where the correspondences are read from or saved to.
		source_image, // The source image
		destination_image, // The destination image
		points); // The detected point correspondences. Each row is of format "x1 y1 x2 y2"
	
	// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
	// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
	std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
	start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation
	neighborhood::GridNeighborhoodGraph neighborhood(&points,
		source_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		source_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.cols / static_cast<double>(cell_number_in_neighborhood_graph_),
		destination_image.rows / static_cast<double>(cell_number_in_neighborhood_graph_),
		cell_number_in_neighborhood_graph_);
	end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
	std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
	printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

	// Checking if the neighborhood graph is initialized successfully.
	if (!neighborhood.isInitialized())
	{
		fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
		return;
	}
	
	// Initializing the samplers
	sampler::UniformSampler main_sampler(&points); // The main sampler is used inside the local optimization
	sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization
	
	// Initializing a visualizer
	progx::MultiHomographyVisualizer visualizer(&points,
		&source_image,
		&destination_image);

	// Applying Progressive-X
	progx::ProgressiveX<neighborhood::GridNeighborhoodGraph,
		DefaultHomographyEstimator,
		sampler::UniformSampler,
		sampler::UniformSampler> progressive_x(&visualizer);

	progressive_x.settings.minimum_number_of_inliers = 8;

	progressive_x.run(points,
		neighborhood,
		main_sampler,
		local_optimization_sampler);

	cv::Mat match_image;
	match_image.create(source_image.rows, // Height
		2 * source_image.cols, // Width
		source_image.type()); // Type
	
	cv::Mat roi_img_result_left =
		match_image(cv::Rect(0, 0, source_image.cols, source_image.rows)); // Img1 will be on the left part
	cv::Mat roi_img_result_right =
		match_image(cv::Rect(destination_image.cols, 0, destination_image.cols, destination_image.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

	cv::Mat roi_image_src = source_image(cv::Rect(0, 0, source_image.cols, source_image.rows));
	cv::Mat roi_image_dst = destination_image(cv::Rect(0, 0, destination_image.cols, destination_image.rows));

	roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
	roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

	for (const auto &inliers : progressive_x.getStatistics().inliers_of_each_model)
	{
		printf("Number of inliers = %d\n", inliers.size());

		cv::Scalar color(255 * static_cast<double>(rand()) / RAND_MAX,
			255 * static_cast<double>(rand()) / RAND_MAX,
			255 * static_cast<double>(rand()) / RAND_MAX);

		// Draw the inlier matches to the images	
		drawMatches(points,
			inliers,
			source_image,
			destination_image,
			match_image,
			10,
			color);
	}

	printf("Press a button to continue...\n");

	// Showing the image
	showImage(match_image,
		"Inlier correspondences",
		1600,
		1200,
		true);
}

void drawMatches(
	const cv::Mat &points_,
	const std::vector<size_t> &inliers_,
	const cv::Mat &image_src_,
	const cv::Mat &image_dst_,
	cv::Mat &out_image_,
	int circle_radius_,
	const cv::Scalar &color_)
{
	for (const auto &idx : inliers_)
	{
		cv::Point2d pt1(points_.at<double>(idx, 0),
			points_.at<double>(idx, 1));
		cv::Point2d pt2(image_dst_.cols + points_.at<double>(idx, 2),
			points_.at<double>(idx, 3));
		
		cv::circle(out_image_, pt1, circle_radius_, color_, static_cast<int>(circle_radius_ * 0.4));
		cv::circle(out_image_, pt2, circle_radius_, color_, static_cast<int>(circle_radius_ * 0.4));
		cv::line(out_image_, pt1, pt2, color_, 2);
	}
}