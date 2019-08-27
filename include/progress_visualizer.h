#pragma once

#include <math.h>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Eigen>

#include "progx_model.h"

namespace progx
{
	class ProgressVisualizer
	{
	protected:
		const std::vector<size_t> * labeling; // The current labeling which assigns each point to a model instance
		size_t label_number, // The number of labels, i.e., model instances
			point_number; // The number of points
		std::vector<cv::Scalar> colors; // The colors which will be used when the points are visualized.

		cv::Scalar getRandomColor()
		{
			return cv::Scalar(255 * static_cast<double>(rand()) / RAND_MAX,
				255 * static_cast<double>(rand()) / RAND_MAX,
				255 * static_cast<double>(rand()) / RAND_MAX);
		}

	public:
		ProgressVisualizer(const size_t point_number_) : 
			point_number(point_number_)
		{

		}

		virtual void visualize(const double &delay_,
			const std::string &process_name_) const = 0;

		void setLabelNumber(size_t label_number_)
		{
			label_number = label_number_;

			if (colors.size() < label_number)
			{
				colors.reserve(label_number);

				// Generate color for the previous outlier instance since it is now a model instance
				colors.back() = getRandomColor();

				// Generate color for every new model instances
				for (size_t i = colors.size(); i < label_number; ++i)
					colors.emplace_back(getRandomColor());

				// The outlier instance should have black color
				colors.back() = cv::Scalar(0, 0, 0);
			}
		}

		void setLabeling(const std::vector<size_t> * labeling_,
			size_t label_number_)
		{
			labeling = labeling_; // The current labeling which assigns each point to a model instance
			label_number = label_number_; // The number of labels, i.e., model instances plus the outlier instance

			// Generate a random for each label. The color will be used
			// when the points are visualized.
			colors.reserve(label_number);
			for (size_t i = 0; i < label_number; ++i)
				colors.emplace_back(getRandomColor());
			// The outlier instance should have black color
			colors.back() = cv::Scalar(0, 0, 0); 
		}

	};

	class MultiHomographyVisualizer : public ProgressVisualizer
	{
	protected:
		const double circe_radius;

		const cv::Mat 
			* const points, // The 2D correspondences
			* const image_source, // The source image
			* const image_destination; // The destination image
		
		void showImage(const cv::Mat &image_,
			const std::string &window_name_,
			const size_t &max_width_,
			const size_t &max_height_,
			const double &delay_) const
		{
			// Resizing the window to fit into the screen if needed
			int window_width = image_.cols,
				window_height = image_.rows;
			if (static_cast<double>(image_.cols) / max_width_ > 1.0 &&
				static_cast<double>(image_.cols) / max_width_ >
				static_cast<double>(image_.rows) / max_height_)
			{
				window_width = max_width_;
				window_height = static_cast<int>(window_width * static_cast<double>(image_.rows) / static_cast<double>(image_.cols));
			}
			else if (static_cast<double>(image_.rows) / max_height_ > 1.0 &&
				static_cast<double>(image_.cols) / max_width_ <
				static_cast<double>(image_.rows) / max_height_)
			{
				window_height = max_height_;
				window_width = static_cast<int>(window_height * static_cast<double>(image_.cols) / static_cast<double>(image_.rows));
			}

			cv::namedWindow(window_name_, CV_WINDOW_NORMAL);
			cv::resizeWindow(window_name_, window_width, window_height);
			cv::imshow(window_name_, image_);
			cv::waitKey(delay_);
		}

	public:
		MultiHomographyVisualizer(
			const cv::Mat * const points_, // The 2D correspondences
			const cv::Mat * const image_source_, // The source image
			const cv::Mat * const image_destination_, // The destination image
			const double circe_radius_ = 10) : // The radius of the circles used for the drawing
			points(points_),
			image_source(image_source_),
			image_destination(image_destination_),
			circe_radius(circe_radius_),
			ProgressVisualizer(points_->rows)
		{

		}

		void visualize(const double &delay_,
			const std::string &process_name_) const
		{
			if (labeling->size() != point_number)
			{
				fprintf(stderr, "A problem occured when doing multi-homography visualization. There are fewer labels than points when visualizing the progress.\n");
				return;
			}

			// Clone the image to do not override them 
			cv::Mat image_source_clone = image_source->clone();
			cv::Mat image_destination_clone = image_destination->clone();

			// Create an image containing both images
			cv::Mat match_image;
			match_image.create(MAX(image_source_clone.rows, image_destination_clone.rows), // Height
				image_source_clone.cols + image_destination_clone.cols, // Width
				image_source_clone.type()); // Type

			// Left the region-of-interest for each image in the big one
			cv::Mat roi_img_result_left =
				match_image(cv::Rect(0, 0, image_source_clone.cols, image_source_clone.rows)); // Img1 will be on the left part
			cv::Mat roi_img_result_right =
				match_image(cv::Rect(image_destination_clone.cols, 0, image_destination_clone.cols, image_destination_clone.rows)); // Img2 will be on the right part, we shift the roi of img1.cols on the right

			// Copy the image regions to the match image
			cv::Mat roi_image_src = 
				image_source_clone(cv::Rect(0, 0, image_source_clone.cols, image_source_clone.rows));
			cv::Mat roi_image_dst = 
				image_destination_clone(cv::Rect(0, 0, image_destination_clone.cols, image_destination_clone.rows));

			roi_image_src.copyTo(roi_img_result_left); //Img1 will be on the left of imgResult
			roi_image_dst.copyTo(roi_img_result_right); //Img2 will be on the right of imgResult

			// Iterate through the points 
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// The color of the current point
				const cv::Scalar &color = 
					colors[labeling->at(point_idx)];

				// The coordinates of the point in the source image
				cv::Point2d point_source(points->at<double>(point_idx, 0),
					points->at<double>(point_idx, 1));
				// The coordinates of the point in the destination image
				cv::Point2d point_destination(image_source_clone.cols + points->at<double>(point_idx, 2),
					points->at<double>(point_idx, 3));

				cv::circle(match_image, point_source, circe_radius, color, -1);
				cv::circle(match_image, point_destination, circe_radius, color, -1);
			}

			// Showing the image
			showImage(match_image,
				process_name_,
				1600,
				1200,
				delay_);
		}
	};
}
