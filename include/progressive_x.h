#pragma once

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "GCRANSAC.h"
#include "types.h"

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
		min_ransac_iteration_number(20),
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
	std::unique_ptr<GCRANSAC<_ModelEstimator, _NeighborhoodGraph>> proposal_engine;
	_ModelEstimator model_estimator;
	MultiModelStatistics statistics;
	std::vector<Model> models;
	std::vector<double> compound_preference_vector;
	double compound_preference_vector_sum,
		compound_preference_vector_length;
	size_t number_of_iterations_without_change;

	void initialize(const cv::Mat &data_);

public:
	MultiModelSettings settings;

	ProgressiveX() 
	{
	}

	void run(const cv::Mat &data_,
		const _NeighborhoodGraph &neighborhood_graph_, // The initialized neighborhood graph
		_MainSampler &main_sampler,
		_LocalOptimizerSampler &local_optimization_sampler);

	const MultiModelStatistics &getStatistics() 
	{
		return statistics;
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

	for (size_t current_iteration = 0; current_iteration < 100; ++current_iteration)
	{
		Model model;

		/**************************
		*** Model proposal step ***
		**************************/
		if (current_iteration > 0)
			proposal_engine->setCompoundModel(&models,
				&compound_preference_vector, 
				compound_preference_vector_sum, 
				compound_preference_vector_length);

		proposal_engine->run(data_,
			model_estimator,
			&main_sampler_,
			&local_optimization_sampler_,
			&neighborhood_graph_,
			model);
		
		const size_t inlier_number = proposal_engine->getRansacStatistics().inliers.size();

		if (inlier_number < minimum_number_of_inliers)
		{
			number_of_iterations_without_change += iteration_number;
			++proposals_without_change;
			if (proposals_without_change == settings.max_proposal_number_without_change)
				break;
			continue;
		}

		statistics.inliers_of_each_model.emplace_back(proposal_engine->getRansacStatistics().inliers);
		break;

	}
}

template<class _NeighborhoodGraph,
	class _ModelEstimator,
	class _MainSampler,
	class _LocalOptimizerSampler>
void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::initialize(const cv::Mat &data_)
{
	// 
	number_of_iterations_without_change = 0;

	// Initializing the proposal engine, i.e., Graph-Cut RANSAC
	proposal_engine = std::make_unique<GCRANSAC<_ModelEstimator, _NeighborhoodGraph>>();
	Settings &proposal_engine_settings = proposal_engine->settings;
	proposal_engine_settings.threshold = settings.inlier_outlier_threshold; // The inlier-outlier threshold
	proposal_engine_settings.spatial_coherence_weight = settings.spatial_coherence_weight; // The weight of the spatial coherence term
	proposal_engine_settings.confidence = settings.confidence; // The required confidence in the results
	proposal_engine_settings.max_local_optimization_number = settings.max_local_optimization_number; // The maximm number of local optimizations
	proposal_engine_settings.max_iteration_number = settings.max_ransac_iteration_number; // The maximum number of iterations
	proposal_engine_settings.min_iteration_number = settings.min_ransac_iteration_number; // The minimum number of iterations
	proposal_engine_settings.neighborhood_sphere_radius = settings.cell_number_in_neighborhood_graph; // The radius of the neighborhood ball
	proposal_engine_settings.core_number = settings.core_number; // The number of parallel processes
}