#pragma once

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "PEARL.h"
#include "GCRANSAC.h"
#include "types.h"
#include "scoring_function_with_compound_model.h"

#include "uniform_sampler.h"
#include "prosac_sampler.h"
#include "progressive_napsac_sampler.h"
#include "fundamental_estimator.h"
#include "homography_estimator.h"
#include "essential_estimator.h"

#include "solver_fundamental_matrix_seven_point.h"
#include "solver_fundamental_matrix_eight_point.h"
#include "solver_homography_four_point.h"
#include "solver_essential_matrix_five_point_stewenius.h"

#include "progx_model.h"

namespace progx
{
	struct MultiModelSettings
	{
		bool do_final_iterated_least_squares, // Flag to decide a final iterated least-squares fitting is needed to polish the output model parameters.
			do_local_optimization, // Flag to decide if local optimization is needed
			do_graph_cut, // Flag to decide of graph-cut is used in the local optimization
			use_inlier_limit; // Flag to decide if an inlier limit is used in the local optimization to speed up the procedure

		size_t minimum_number_of_inliers,
			max_proposal_number_without_change,
			cell_number_in_neighborhood_graph,
			max_local_optimization_number, // Maximum number of local optimizations
			min_iteration_number_before_lo, // Minimum number of RANSAC iterations before applying local optimization
			min_ransac_iteration_number, // Minimum number of RANSAC iterations
			max_ransac_iteration_number, // Maximum number of RANSAC iterations
			max_unsuccessful_model_generations, // Maximum number of unsuccessful model generations
			max_least_squares_iterations, // Maximum number of iterated least-squares iterations
			max_graph_cut_number, // Maximum number of graph-cuts applied in each current_iteration
			core_number; // Number of parallel threads

		double confidence, // Required confidence in the result
			neighborhood_sphere_radius, // The radius of the ball used for creating the neighborhood graph
			inlier_outlier_threshold, // The inlier-outlier threshold
			spatial_coherence_weight; // The weight of the spatial coherence term

		MultiModelSettings() :
			minimum_number_of_inliers(0),
			do_final_iterated_least_squares(true),
			do_local_optimization(true),
			do_graph_cut(true),
			use_inlier_limit(false),
			cell_number_in_neighborhood_graph(8),
			max_local_optimization_number(20),
			max_proposal_number_without_change(10),
			max_graph_cut_number(std::numeric_limits<size_t>::max()),
			max_least_squares_iterations(20),
			min_iteration_number_before_lo(20),
			min_ransac_iteration_number(200),
			neighborhood_sphere_radius(20),
			max_ransac_iteration_number(std::numeric_limits<size_t>::max()),
			max_unsuccessful_model_generations(100),
			core_number(1),
			spatial_coherence_weight(0.14),
			inlier_outlier_threshold(2.0),
			confidence(0.95)
		{

		}
	};

	struct MultiModelStatistics
	{
		double processing_time;
		std::vector<std::vector<size_t>> inliers_of_each_model;
	};

	template<class _NeighborhoodGraph,
		class _ModelEstimator,
		class _MainSampler,
		class _LocalOptimizerSampler>
		class ProgressiveX
	{
	protected:
		std::unique_ptr<gcransac::GCRANSAC<_ModelEstimator,
			_NeighborhoodGraph,
			MSACScoringFunctionWithCompoundModel<_ModelEstimator>>> proposal_engine;
		std::unique_ptr<pearl::PEARL<_ModelEstimator,
			_NeighborhoodGraph>> model_optimizer;
		_ModelEstimator model_estimator;
		MultiModelStatistics statistics;
		std::vector<Model<_ModelEstimator>> models;
		Eigen::MatrixXd compound_preference_vector;
		double compound_preference_vector_sum,
			compound_preference_vector_length,
			truncated_squared_threshold;
		size_t number_of_iterations_without_change,
			point_number;

		void initialize(const cv::Mat &data_);

		inline bool isPutativeModelValid(
			const cv::Mat &data_,
			const progx::Model<_ModelEstimator> &model_,
			const gcransac::RANSACStatistics &statistics_);

		void updateCompoundModel(const cv::Mat &data_);

	public:
		MultiModelSettings settings;

		ProgressiveX()
		{
		}

		void run(const cv::Mat &data_,
			const _NeighborhoodGraph &neighborhood_graph_, // The initialized neighborhood graph
			_MainSampler &main_sampler,
			_LocalOptimizerSampler &local_optimization_sampler);

		const MultiModelStatistics &getStatistics() const
		{
			return statistics;
		}

		MultiModelStatistics &getMutableStatistics()
		{
			return statistics;
		}

		const std::vector<Model<_ModelEstimator>> &getModels() const
		{
			return models;
		}

		std::vector<Model<_ModelEstimator>> &getMutableModels()
		{
			return models;
		}

		size_t getModelNumber() const
		{
			return models.size();
		}

	};

	template<class _NeighborhoodGraph,
		class _ModelEstimator,
		class _MainSampler,
		class _LocalOptimizerSampler>
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::run(
		const cv::Mat &data_,
		const _NeighborhoodGraph &neighborhood_graph_, // The initialized neighborhood graph
		_MainSampler &main_sampler_,
		_LocalOptimizerSampler &local_optimization_sampler_)
	{
		// Initializing the procedure
		initialize(data_);

		size_t proposals_without_change = 0;

		for (size_t current_iteration = 0; current_iteration < 10; ++current_iteration)
		{
			/***********************************
			*** Model instance proposal step ***
			***********************************/		   
			// The putative model proposed by the proposal engine
			progx::Model<_ModelEstimator> putative_model;

			// Applying the proposal engine to get a new putative model
			proposal_engine->run(data_, // The data points
				model_estimator, // The model estimator to be used
				&main_sampler_, // The sampler used for the main RANSAC loop
				&local_optimization_sampler_, // The samplre used for the local optimization
				&neighborhood_graph_, // The neighborhood graph
				putative_model);

			// Set a reference to the model estimator in the putative model instance
			putative_model.setEstimator(&model_estimator);
						
			// Get the RANSAC statistics to know the inliers of the proposal
			const gcransac::RANSACStatistics &proposal_engine_statistics = 
				proposal_engine->getRansacStatistics();

			/*************************************
			*** Model instance validation step ***
			*************************************/
			if (!isPutativeModelValid(data_,
				putative_model,
				proposal_engine_statistics))
			{
				number_of_iterations_without_change += proposal_engine_statistics.iteration_number;
				++proposals_without_change;
				if (proposals_without_change == settings.max_proposal_number_without_change)
					break;
				continue;
			}

			/******************************************
			*** Compound instance optimization step ***
			******************************************/
			// Add the putative instance to the compound one
			models.emplace_back(putative_model);



			statistics.inliers_of_each_model.emplace_back(proposal_engine->getRansacStatistics().inliers);

			// Update the compound model
			updateCompoundModel(data_);
		}
	}

	template<class _NeighborhoodGraph,
		class _ModelEstimator,
		class _MainSampler,
		class _LocalOptimizerSampler>
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::initialize(const cv::Mat &data_)
	{
		// 
		point_number = data_.rows;
		number_of_iterations_without_change = 0;
		truncated_squared_threshold = std::pow(3.0 / 2.0 * settings.inlier_outlier_threshold, 2);
		compound_preference_vector = Eigen::MatrixXd::Zero(data_.rows, 1);

		// Initializing the model optimizer, i.e., PEARL
		model_optimizer = std::make_unique<pearl::PEARL<_ModelEstimator,
			_NeighborhoodGraph>>();

		// Initializing the proposal engine, i.e., Graph-Cut RANSAC
		proposal_engine = std::make_unique < gcransac::GCRANSAC <_ModelEstimator,
			_NeighborhoodGraph,
			MSACScoringFunctionWithCompoundModel<_ModelEstimator>>>();

		Settings &proposal_engine_settings = proposal_engine->settings;
		proposal_engine_settings.threshold = settings.inlier_outlier_threshold; // The inlier-outlier threshold
		proposal_engine_settings.spatial_coherence_weight = settings.spatial_coherence_weight; // The weight of the spatial coherence term
		proposal_engine_settings.confidence = settings.confidence; // The required confidence in the results
		proposal_engine_settings.max_local_optimization_number = settings.max_local_optimization_number; // The maximm number of local optimizations
		proposal_engine_settings.max_iteration_number = settings.max_ransac_iteration_number; // The maximum number of iterations
		proposal_engine_settings.min_iteration_number = settings.min_ransac_iteration_number; // The minimum number of iterations
		proposal_engine_settings.neighborhood_sphere_radius = settings.cell_number_in_neighborhood_graph; // The radius of the neighborhood ball
		proposal_engine_settings.core_number = settings.core_number; // The number of parallel processes

		MSACScoringFunctionWithCompoundModel<_ModelEstimator> &scoring =
			proposal_engine->getMutableScoringFunction();
		scoring.setCompoundModel(&models, 
			&compound_preference_vector);
	}
	
	template<class _NeighborhoodGraph,
		class _ModelEstimator,
		class _MainSampler,
		class _LocalOptimizerSampler>
	inline bool ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::isPutativeModelValid(
		const cv::Mat &data_,
		const progx::Model<_ModelEstimator> &model_,
		const gcransac::RANSACStatistics &statistics_)
	{
		// Number of inliers without considering that there are more model instances in the scene
		const size_t inlier_number = statistics_.inliers.size();

		// If the putative model has fewer inliers than the minimum, it is considered invalid.
		if (inlier_number < MAX(_ModelEstimator::sampleSize(), settings.minimum_number_of_inliers))
			return false;

		return true;
	}


	template<class _NeighborhoodGraph,
		class _ModelEstimator,
		class _MainSampler,
		class _LocalOptimizerSampler>
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::updateCompoundModel(const cv::Mat &data_)
	{
		// Do not do anything if there are no models in the compound instance
		if (models.size() == 0)
			return;

		// Reset the preference vector of the compound instance
		compound_preference_vector.setConstant(0);

		// Iterate through all instances in the compound one and 
		// update the preference values
		for (auto &model : models)
		{
			// Initialize the model's preference vector if needed
			if (model.preference_vector.rows() != point_number ||
				model.preference_vector.cols() != 1)
				model.preference_vector.resize(point_number, 1);

			// Iterate through all points and estimate the preference values
			double squared_residual;
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// The point-to-model residual
				squared_residual = model_estimator.squaredResidual(data_.row(point_idx), model);
				
				// Update the preference vector of the current model since it might be changed
				// due to the optimization.
				model.preference_vector(point_idx) = 
					MAX(0, 1.0 - squared_residual / truncated_squared_threshold);

				// Update the preference vector of the compound model. Since the point-to-<compound model>
				// residual is defined through the union of distance fields of the contained models,
				// the implied preference is the highest amongst the stored model instances. 
				compound_preference_vector(point_idx) =
					MAX(compound_preference_vector(point_idx), model.preference_vector(point_idx));
			}
		}
	}
}