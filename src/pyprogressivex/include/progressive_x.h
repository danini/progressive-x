#pragma once

#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "PEARL.h"
#include "GCRANSAC.h"
#include "types.h"
#include "scoring_function_with_compound_model.h"
#include "progress_visualizer.h"

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
		// The settings of the proposal engine
		gcransac::utils::Settings proposal_engine_settings;

		size_t minimum_number_of_inliers,
			max_proposal_number_without_change,
			cell_number_in_neighborhood_graph,
			maximum_model_number_to_optimize;

		int maximum_model_number;

		double maximum_tanimoto_similarity,
			proposal_engine_confidence,
			minimum_triangle_size,
			confidence, // Required confidence in the result
			one_minus_confidence, // 1 - confidence
			inlier_outlier_threshold, // The inlier-outlier threshold
			spatial_coherence_weight; // The weight of the spatial coherence term

		void setConfidence(const double &confidence_)
		{
			confidence = confidence_;
			one_minus_confidence = 1.0 - confidence;
		}

		MultiModelSettings() :
			maximum_tanimoto_similarity(0.5),
			minimum_triangle_size(100),
			minimum_number_of_inliers(20),
			cell_number_in_neighborhood_graph(8),
			max_proposal_number_without_change(10),
			spatial_coherence_weight(0.14),
			inlier_outlier_threshold(2.0),
			confidence(0.95),
			proposal_engine_confidence(1.0),
			one_minus_confidence(0.05),
			maximum_model_number_to_optimize(3),
			maximum_model_number(std::numeric_limits<size_t>::max())
		{
			proposal_engine_settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph;
			proposal_engine_settings.max_iteration_number = 5000;
			proposal_engine_settings.max_local_optimization_number = 50;
			proposal_engine_settings.threshold = inlier_outlier_threshold;
			proposal_engine_settings.confidence = confidence;
			proposal_engine_settings.spatial_coherence_weight = 0.975;
		}
	};

	struct IterationStatistics
	{
		double time_of_proposal_engine,
			time_of_model_validation,
			time_of_optimization,
			time_of_compound_model_update;
		size_t number_of_instances;
	};

	struct MultiModelStatistics
	{
		double processing_time,
			total_time_of_proposal_engine,
			total_time_of_model_validation,
			total_time_of_optimization,
			total_time_of_compound_model_calculation;
		std::vector<std::vector<size_t>> inliers_of_each_model;
		std::vector<size_t> labeling;
		std::vector<double> scores;
		std::vector<IterationStatistics> iteration_statistics;

		void addIterationStatistics(IterationStatistics iteration_statistics_)
		{
			iteration_statistics.emplace_back(iteration_statistics_);

			total_time_of_proposal_engine += iteration_statistics_.time_of_proposal_engine;
			total_time_of_model_validation += iteration_statistics_.time_of_model_validation;
			total_time_of_optimization += iteration_statistics_.time_of_optimization;
			total_time_of_compound_model_calculation += iteration_statistics_.time_of_compound_model_update;
		}
	};

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
		class ProgressiveX
	{
	protected:
		// The proposal engine (i.e. Graph-Cut RANSAC) estimating a putative model in the beginning of each iteration
		std::unique_ptr<gcransac::GCRANSAC<
			// The model estimator used for estimating the instance parameters from a set of points
			_ModelEstimator,
			// The type of the used neighborhood graph
			_NeighborhoodGraph,
			// The scoring class which consideres the compound instance when calculating the score of a model instance
			EPOSScoringFunctionWithCompoundModel<_ModelEstimator>>> 
			proposal_engine;

		// The model optimizer optimizing the compound model parameters in each iteration
		std::unique_ptr<pearl::PEARL<
			// The type of the used neighborhood graph which is needed for determining the spatial coherence cost inside PEARL
			_NeighborhoodGraph,
			// The model estimator used for estimating the instance parameters from a set of points
			_ModelEstimator>> model_optimizer;

		// The model estimator which estimates the model parameters from a set of points
		_ModelEstimator *model_estimator;
		
		// Verbose
		bool verbose;
	
		// The statistics of Progressive-X containing everything which the user might be curious about,
		// e.g., processing time, results, etc.
		MultiModelStatistics statistics;

		// The set of models (i.e., the compound instance) maintained throughout the multi-model fitting. 
		std::vector<progx::Model<_ModelEstimator>> models;

		// The preference vector of the compound model instance
		Eigen::VectorXd compound_preference_vector;

		// The truncated squared inlier-outlier threshold
		double truncated_squared_threshold,
			scoring_exponent; 

		size_t point_number; // The number of points

		// The visualizer which demonstrates the procedure by showing the labeling in each intermediate step
		ProgressVisualizer * const visualizer;

		// The settings of the algorithm
		MultiModelSettings settings;

		// Setting all the initial parameters
		void initialize(const cv::Mat &data_);

		// Check if the putative model instance should be included in the optimization procedure
		inline bool isPutativeModelValid(
			const cv::Mat &data_, // All data points
			progx::Model<_ModelEstimator> &model_, // The model instance to check
			const gcransac::utils::RANSACStatistics &statistics_); // The RANSAC statistics of the putative model

		// Update the compound model's preference vector
		void updateCompoundModel(const cv::Mat &data_);
		
		// Predicts the number of inliers in the data which are unseen yet, 
		// i.e. not covered by the compound instance.
		size_t getPredictedUnseenInliers(
			const double confidence_, // The RANSAC confidence
			const size_t sample_size_, // The size of a minimal sample
			const size_t iteration_number_, // The current number of RANSAC iterations
			const size_t inlier_number_of_compound_model_); // The number of inliers of the compound model instance

		// The function applying Progressive-X to a set of data points
		void spedUpFitting(const cv::Mat &data_, // All data points
			const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
			_ModelEstimator &model_estimator_, // The used model estimator
			_MainSampler &main_sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
			_LocalOptimizerSampler &local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC

	public:

		ProgressiveX(ProgressVisualizer * const visualizer_ = nullptr) :
			visualizer(visualizer_),
			scoring_exponent(2),
			verbose(false)
		{
		}

		// The function applying Progressive-X to a set of data points
		void run(const cv::Mat &data_, // All data points
			const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
			_ModelEstimator &model_estimator_, // The used model estimator
			_MainSampler &main_sampler, // The sampler used in the main RANSAC loop of GC-RANSAC
			_LocalOptimizerSampler &local_optimization_sampler); // The sampler used in the local optimization of GC-RANSAC
		
		void setScoringExponent(const double &scoring_exponent_)
		{
			scoring_exponent = scoring_exponent_;
		}

		void log(bool do_logging_)
		{
			verbose = do_logging_;
		}

		// Returns a constant reference to the settings of the multi-model fitting
		const MultiModelSettings &getSettings() const
		{
			return settings;
		}

		// Returns a reference to the settings of the multi-model fitting
		MultiModelSettings &getMutableSettings()
		{
			return settings;
		}


		// Returns a constant reference to the statistics of the multi-model fitting
		const MultiModelStatistics &getStatistics() const
		{
			return statistics;
		}

		// Returns a reference to the statistics of the multi-model fitting
		MultiModelStatistics &getMutableStatistics()
		{
			return statistics;
		}

		// Returns a constant reference to the estimated model instances
		const std::vector<Model<_ModelEstimator>> &getModels() const
		{
			return models;
		}

		// Returns a reference to the statistics of the multi-model fitting
		std::vector<Model<_ModelEstimator>> &getMutableModels()
		{
			return models;
		}

		// Returns the number of model instances estimated
		size_t getModelNumber() const
		{
			return models.size();
		}
	};


	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
		void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::spedUpFitting(
			const cv::Mat &data_, // All data points
			const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
			_ModelEstimator &model_estimator_,
			_MainSampler &main_sampler_, // The sampler used in the main RANSAC loop of GC-RANSAC
			_LocalOptimizerSampler &local_optimization_sampler_) // The sampler used in the local optimization of GC-RANSAC
	{
		size_t number_of_ransac_iterations = 0; 
		cv::Mat current_data = data_.clone();
		std::vector<size_t> indices(data_.rows);
		for (size_t i = 0; i < data_.rows; ++i)
			indices[i] = i;
		std::chrono::time_point<std::chrono::system_clock> start, end, // Variables for time measurement
			main_start = std::chrono::system_clock::now(), main_end;

		for (size_t current_iteration = 0; current_iteration < settings.maximum_model_number; ++current_iteration)
		{
			// Initializing the proposal engine, i.e., Graph-Cut RANSAC
			gcransac::GCRANSAC <_ModelEstimator, _NeighborhoodGraph, gcransac::EPOSScoringFunction<_ModelEstimator>>
				local_proposal_engine;
			
			// Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
			// in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
			gcransac::neighborhood::FlannNeighborhoodGraph neighborhood(&current_data, // All data points
				20.0); // The radius of the neighborhood ball for determining the neighborhoods.

			local_proposal_engine.settings = settings.proposal_engine_settings;
			local_proposal_engine.settings.confidence = 1.0;
			local_proposal_engine.settings.threshold = settings.inlier_outlier_threshold;
			local_proposal_engine.settings.spatial_coherence_weight = settings.spatial_coherence_weight;

			if (verbose)
			{
				printf("-------------------------------------------\n");
				printf("Iteration %d.\n", current_iteration + 1);
			}

			// Statistics of the current iteration
			IterationStatistics iteration_statistics;

			/***********************************
			*** Model instance proposal step ***
			***********************************/
			// The putative model proposed by the proposal engine
			progx::Model<_ModelEstimator> putative_model;

			// Reset the samplers
			/*gcransac::sampler::ProsacSampler main_sampler(&current_data, // The data points
				_ModelEstimator::sampleSize()); // The size of a minimal sample*/
			_MainSampler main_sampler(&current_data);
			_LocalOptimizerSampler local_optimization_sampler(&current_data);
						
			// Applying the proposal engine to get a new putative model
			local_proposal_engine.run(current_data, // The data points
				*model_estimator, // The model estimator to be used
				&main_sampler, // The sampler used for the main RANSAC loop
				&local_optimization_sampler, // The sampler used for the local optimization
				&neighborhood, // The neighborhood graph
				putative_model);

			if (putative_model.descriptor.rows() == 0 ||
				putative_model.descriptor.cols() == 0)
				continue;

			// Add the putative instance to the compound one
			models.emplace_back(putative_model);

			// Get the RANSAC statistics to know the inliers of the proposal
			const gcransac::utils::RANSACStatistics &proposal_engine_statistics =
				local_proposal_engine.getRansacStatistics();

			statistics.scores.emplace_back(proposal_engine_statistics.score);

			// Update the current iteration's statistics
			iteration_statistics.time_of_proposal_engine =
				proposal_engine_statistics.processing_time;

			// Increase the total number of RANSAC iterations
			number_of_ransac_iterations +=
				proposal_engine_statistics.iteration_number;

			if (verbose)
			{
				printf("A model proposed with %d inliers in %f seconds (%d iterations).\n",
					proposal_engine_statistics.inliers.size(),
					iteration_statistics.time_of_proposal_engine,
					proposal_engine_statistics.iteration_number);
			}

			if (proposal_engine_statistics.inliers.size() < settings.minimum_number_of_inliers)
				continue;

			/******************************************
			*** Remove the putative model's inliers ***
			******************************************/
			std::vector<bool> mask(current_data.rows, 0);
			statistics.inliers_of_each_model.emplace_back(std::vector<size_t>());
			for (const size_t &inlier_idx : proposal_engine_statistics.inliers)
			{
				mask[inlier_idx] = 1;
				statistics.inliers_of_each_model.back().emplace_back(indices[inlier_idx]);
			}

			cv::Mat temp_data = current_data.clone();
			std::vector<size_t> old_indices = indices;
			indices.resize(0);
			current_data = cv::Mat(current_data.rows - proposal_engine_statistics.inliers.size(), current_data.cols, CV_64F);
			size_t current_row = 0;
			for (size_t r = 0; r < mask.size(); ++r)
			{
				if (!mask[r])
				{
					temp_data.row(r).copyTo(current_data.row(current_row));
					indices.emplace_back(old_indices[r]);
					++current_row;
				}
			}
		}

		main_end = std::chrono::system_clock::now();

		// The elapsed time in seconds
		std::chrono::duration<double> elapsed_seconds; // The elapsed time in seconds
		elapsed_seconds = main_end - main_start;

		statistics.processing_time = elapsed_seconds.count();
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::run(
		const cv::Mat &data_, // All data points
		const _NeighborhoodGraph &neighborhood_graph_, // The neighborhood graph
		_ModelEstimator &model_estimator_,
		_MainSampler &main_sampler_, // The sampler used in the main RANSAC loop of GC-RANSAC
		_LocalOptimizerSampler &local_optimization_sampler_) // The sampler used in the local optimization of GC-RANSAC
	{
		// Initializing the procedure
		initialize(data_);

		size_t number_of_ransac_iterations = 0, // The total number of RANSAC iterations
			unaccepted_putative_instances = 0, // The number of consecutive putative instances not accepted to be optimized
			unseen_inliers = point_number; // Predicted number of unseen inliers in the data		
		std::chrono::time_point<std::chrono::system_clock> start, end, // Variables for time measurement
			main_start = std::chrono::system_clock::now(), main_end;
		std::chrono::duration<double> elapsed_seconds; // The elapsed time in seconds
		model_estimator = &model_estimator_;

		if (verbose)
			printf("The main iteration is started...\n");

		if (settings.maximum_model_number_to_optimize < settings.maximum_model_number)
		{
			spedUpFitting(data_,
				neighborhood_graph_,
				model_estimator_,
				main_sampler_,
				local_optimization_sampler_);
			return;
		}

		for (size_t current_iteration = 0; current_iteration < settings.maximum_model_number < 0 ? 20 : 2 * settings.maximum_model_number; ++current_iteration)
		{
			if (verbose)
			{			
				printf("-------------------------------------------\n");
				printf("Iteration %d.\n", current_iteration + 1);
			}

			if (current_iteration > 100)
				break;

			// Statistics of the current iteration
			IterationStatistics iteration_statistics;

			/***********************************
			*** Model instance proposal step ***
			***********************************/	
			// The putative model proposed by the proposal engine
			progx::Model<_ModelEstimator> putative_model;

			// Reset the samplers
			main_sampler_.reset();
			local_optimization_sampler_.reset();

			// Applying the proposal engine to get a new putative model
			proposal_engine->run(data_, // The data points
				*model_estimator, // The model estimator to be used
				&main_sampler_, // The sampler used for the main RANSAC loop
				&local_optimization_sampler_, // The sampler used for the local optimization
				&neighborhood_graph_, // The neighborhood graph
				putative_model);
			
			if (putative_model.descriptor.rows() == 0 ||
				putative_model.descriptor.cols() == 0)
				continue;

			// Set a reference to the model estimator in the putative model instance
			putative_model.setEstimator(model_estimator);
						
			// Get the RANSAC statistics to know the inliers of the proposal
			const gcransac::utils::RANSACStatistics &proposal_engine_statistics = 
				proposal_engine->getRansacStatistics();

			// Update the current iteration's statistics
			iteration_statistics.time_of_proposal_engine = 
				proposal_engine_statistics.processing_time;

			// Increase the total number of RANSAC iterations
			number_of_ransac_iterations += 
				proposal_engine_statistics.iteration_number;

			if (verbose)
			{			
				printf("A model proposed with %d inliers\nin %f seconds (%d iterations).\n",
					proposal_engine_statistics.inliers.size(),
					iteration_statistics.time_of_proposal_engine,
					proposal_engine_statistics.iteration_number);
			}
				

			/*************************************
			*** Model instance validation step ***
			*************************************/
			if (verbose)
				printf("Check if the model should be added to the compound instance.\n");
				
			// The starting time of the model validation
			start = std::chrono::system_clock::now();
			if (!isPutativeModelValid(data_,
				putative_model,
				proposal_engine_statistics))
			{
				if (verbose)	
					printf("The model is not accepted to be added to the compound instances. The number of consecutively rejected proposals is %d (< %d)\n",
						unaccepted_putative_instances,
						settings.max_proposal_number_without_change);
					
				++unaccepted_putative_instances;
				if (unaccepted_putative_instances == settings.max_proposal_number_without_change)
					break;
				continue;
			}
			
			// The end time of the model validation
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_model_validation =
				elapsed_seconds.count();

			if (verbose)	
				printf("The model has been accepted in %f seconds.\n", iteration_statistics.time_of_model_validation);

			/******************************************
			*** Compound instance optimization step ***
			******************************************/
			// The starting time of the model optimization
			start = std::chrono::system_clock::now();

			// Add the putative instance to the compound one
			models.emplace_back(putative_model);

			// If only a single model instance is known, use the inliers of GC-RANSAC
			// to initialize the labeling.

			if (verbose)	
				printf("Model optimization started...\n");
			if (models.size() == 1)
			{
				// Store the inliers of the current model to the statistics object
				statistics.inliers_of_each_model.emplace_back(
					proposal_engine->getRansacStatistics().inliers);

				// Set the labeling so that the outliers will have label 1 and the inliers label 0.
				std::fill(statistics.labeling.begin(), statistics.labeling.end(), 1);
				for (const size_t &point_idx : statistics.inliers_of_each_model.back())
					statistics.labeling[point_idx] = 0;
			}
			// Otherwise, apply an optimizer the determine the labeling
			else
			{
				// Apply the model optimizer
				model_optimizer->run(data_,
					&neighborhood_graph_,
					model_estimator,
					&models);

				size_t model_number = 0;
				model_optimizer->getLabeling(statistics.labeling, model_number);

				if (verbose && 
					model_number != models.size())
				{
					printf("Models have been removed during the optimization.\n");
				}
			}

			// The end time of the model optimization
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_optimization =
				elapsed_seconds.count();

			if (verbose)
				printf("Model optimization finished in %f seconds.\n", iteration_statistics.time_of_optimization);

			// The starting time of the model validation
			start = std::chrono::system_clock::now();

			// Update the compound model
			updateCompoundModel(data_);

			// The end time of the model optimization
			end = std::chrono::system_clock::now();

			// The elapsed time in seconds
			elapsed_seconds = end - start;

			// Update the current iteration's statistics
			iteration_statistics.time_of_compound_model_update =
				elapsed_seconds.count();

			if (verbose)
				printf("Compound instance is updated in %f seconds.\n", iteration_statistics.time_of_compound_model_update);

			// Store the instance number in the iteration's statistics
			iteration_statistics.number_of_instances = models.size();

			/************************************
			*** Updating the iteration number ***
			************************************/
			// If there is a only a single model instance, PEARL was not applied. 
			// Thus, the inliers are in the RANSAC statistics of the instance.
			if (models.size() == 1)
				unseen_inliers = getPredictedUnseenInliers(settings.one_minus_confidence, // 1.0 - confidence
					_ModelEstimator::sampleSize(), // The size of a minimal sample
					number_of_ransac_iterations, // The total number of RANSAC iterations applied
					statistics.inliers_of_each_model.size()); // The inlier number of the compound model instance

			else
				unseen_inliers = getPredictedUnseenInliers(settings.one_minus_confidence, // 1.0 - confidence
					_ModelEstimator::sampleSize(), // The size of a minimal sample
					number_of_ransac_iterations, // The total number of RANSAC iterations applied
					point_number - model_optimizer->getOutlierNumber()); // The inlier number of the compound model instance

			// Add the current iteration's statistics to the statistics object
			statistics.addIterationStatistics(iteration_statistics);

			if (verbose)
				printf("The predicted number of inliers (with confidence %f)\nnot covered by the compound instance is %d.", settings.confidence, unseen_inliers);

			// If it is likely, that there are fewer inliers in the data than the minimum number,
			// terminate.
			if (unseen_inliers < settings.minimum_number_of_inliers)
				break;

			// If we have enough models, terminate.
			if (getModelNumber() >= settings.maximum_model_number)
				break;

			// Visualize the labeling results if needed
			if (visualizer != nullptr)
			{
				visualizer->setLabelNumber(models.size() + 1);
				visualizer->visualize(0, "Labeling");
			}
		}

		main_end = std::chrono::system_clock::now();

		// The elapsed time in seconds
		elapsed_seconds = main_end - main_start;

		statistics.processing_time = elapsed_seconds.count();
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	size_t ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::getPredictedUnseenInliers(
		const double one_minus_confidence_,
		const size_t sample_size_,
		const size_t iteration_number_,
		const size_t inlier_number_of_compound_model_)
	{
		// Number of points in the data which have not yet been assigned to any model
		const size_t unseen_point_number = point_number - inlier_number_of_compound_model_;

		const double one_over_iteration_number = 1.0 / iteration_number_;
		const double one_over_sample_size = 1.0 / sample_size_;

		// Calculate the ratio of the maximum inlier number from the sample size, current iteration number and confidence
		const double inlier_ratio = 
			pow(1.0 - pow(one_minus_confidence_, one_over_iteration_number), one_over_sample_size);

		// Return the number of unseen inliers
		return static_cast<size_t>(std::round(unseen_point_number * inlier_ratio));
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::initialize(const cv::Mat &data_)
	{
		point_number = data_.rows; // The number of data points
		statistics.labeling.resize(point_number, 0); // The labeling which assigns each point to a model instance. Initially, all points are considered outliers.
		truncated_squared_threshold = 9.0 / 4.0 * settings.inlier_outlier_threshold *  settings.inlier_outlier_threshold;
		compound_preference_vector = Eigen::VectorXd::Zero(data_.rows);

		// Initializing the model optimizer, i.e., PEARL
		model_optimizer = std::make_unique<pearl::PEARL<_NeighborhoodGraph,
			_ModelEstimator>>(
				settings.inlier_outlier_threshold,
				settings.spatial_coherence_weight,
				settings.minimum_number_of_inliers);
		model_optimizer->log(verbose);
		
		// Initializing the proposal engine, i.e., Graph-Cut RANSAC
		proposal_engine = std::make_unique < gcransac::GCRANSAC <_ModelEstimator,
			_NeighborhoodGraph,
			EPOSScoringFunctionWithCompoundModel<_ModelEstimator>>>();

		gcransac::utils::Settings &proposal_engine_settings = proposal_engine->settings;
		proposal_engine_settings = settings.proposal_engine_settings;
		proposal_engine_settings.confidence = settings.proposal_engine_confidence;
		proposal_engine_settings.threshold = settings.inlier_outlier_threshold;
		proposal_engine_settings.spatial_coherence_weight = settings.spatial_coherence_weight;

		EPOSScoringFunctionWithCompoundModel<_ModelEstimator> &scoring =
			proposal_engine->getMutableScoringFunction();
		scoring.setCompoundModel(&models, 
			&compound_preference_vector);
		scoring.setExponent(scoring_exponent);

		// Initialize the visualizer if needed
		if (visualizer != nullptr)
		{
			visualizer->setLabeling(&statistics.labeling, // Set the labeling pointer 
				1); // Initially, only the outlier model instance exists
		}
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	inline bool ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::isPutativeModelValid(
		const cv::Mat &data_,
		progx::Model<_ModelEstimator> &model_,
		const gcransac::utils::RANSACStatistics &statistics_)
	{
		// Number of inliers without considering that there are more model instances in the scene
		const size_t inlier_number = statistics_.inliers.size();

		// If the putative model has fewer inliers than the minimum, it is considered invalid.
		if (inlier_number < MAX(_ModelEstimator::sampleSize(), settings.minimum_number_of_inliers))
			return false;

		// Calculate the preference vector of the current model
		model_.setPreferenceVector(data_, // All data points
			truncated_squared_threshold); // The truncated squared threshold

		// Calculate the Tanimoto-distance of the preference vectors of the current and the 
		// compound model instance.
		const double dot_product = model_.preference_vector.dot(compound_preference_vector);
		double tanimoto_similarity = dot_product / 
			(model_.preference_vector.squaredNorm() + compound_preference_vector.squaredNorm() - dot_product);

		if (settings.maximum_tanimoto_similarity < tanimoto_similarity)
			return false;

		return true;
	}

	template<class _NeighborhoodGraph, // The type of the used neighborhood graph
		class _ModelEstimator, // The model estimator used for estimating the instance parameters from a set of points
		class _MainSampler, // The sampler used in the main RANSAC loop of GC-RANSAC
		class _LocalOptimizerSampler> // The sampler used in the local optimization of GC-RANSAC
	void ProgressiveX<_NeighborhoodGraph, _ModelEstimator, _MainSampler, _LocalOptimizerSampler>::updateCompoundModel(const cv::Mat &data_)
	{
		// Do not do anything if there are no models in the compound instance
		if (models.size() == 0)
			return;

		// Reset the preference vector of the compound instance
		compound_preference_vector.setConstant(0);

		// Updating the score
		statistics.scores.resize(models.size());

		// Iterate through all instances in the compound one and 
		// update the preference values
		size_t model_idx = 0;
		for (auto &model : models)
		{
			// Initialize the score
			statistics.scores[model_idx] = 0.0;

			// Iterate through all points and estimate the preference values
			double squared_residual;
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				// The point-to-model residual
				squared_residual = model_estimator->squaredResidual(data_.row(point_idx), model);
				
				// Update the preference vector of the compound model. Since the point-to-<compound model>
				// residual is defined through the union of distance fields of the contained models,
				// the implied preference is the highest amongst the stored model instances. 
				compound_preference_vector(point_idx) =
					MAX(compound_preference_vector(point_idx), model.preference_vector(point_idx));

				// Updating the score
				statistics.scores[model_idx] +=
					model.preference_vector(point_idx);
			}
			++model_idx;
		}
	}
}