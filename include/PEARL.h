#pragma once

#include <math.h>
#include <random>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Eigen>

#include "GCoptimization.h"
#include "progx_model.h"

namespace pearl
{
	template<class _ModelEstimator>
	struct EnergyDataStructure
	{
		const cv::Mat &points;
		const std::vector<progx::Model<_ModelEstimator>> * const model_instances;
		const double spatial_coherence_weight,
			inlier_outlier_threshold;
		std::vector<std::vector<double>> residuals;

		EnergyDataStructure(const cv::Mat &points_,
			const std::vector<progx::Model<_ModelEstimator>> * const model_instances_,
			const double spatial_coherence_weight_,
			const double inlier_outlier_threshold_) :
			points(points_),
			model_instances(model_instances_),
			spatial_coherence_weight(spatial_coherence_weight_),
			inlier_outlier_threshold(inlier_outlier_threshold_)
		{
			residuals.resize(_points.rows, 
				std::vector<double>(model_instances_->size(), -1));
		}
	};

	inline double spatialCoherenceEnergyFunctor(int p1, int p2, int l1, int l2, void *data)
	{
		/*EnergyDataStructure *myData = (EnergyDataStructure *)data;
		float lambda = myData->energy_lambda;
		return (l1 != l2) ? lambda : 0;*/
		return 0;
	}

	inline double dataEnergyFunctor(int p, int l, void *data)
	{
		return 0;

		/*EnergyDataStructure *myData = (EnergyDataStructure *)data;
		float lambda = myData->energy_lambda;

		float threshold = myData->threshold;
		float sqr_threshold = threshold * threshold;
		float truncated_threshold = sqr_threshold * 9 / 4;

		if (l == 0)
			return (1 - lambda) * truncated_threshold;

		vector<vector<float>> *distances = &myData->distances;

		float distance = -1; // distances->at(p)[l - 1];
		if (distance == -1)
		{
			Model instance = myData->instances->at(l - 1);
			cv::Mat point = myData->points.row(p);
			distance = instance.estimator->Error(point, instance);
			distance = distance * distance;

			//distances->at(p)[l - 1] = distance;
		}

		if (distance > truncated_threshold)
			return 2 * (1 - lambda) * truncated_threshold;
		return (1 - lambda) * distance;*/
	}

	template<class _NeighborhoodGraph,
		class _ModelEstimator>
	class PEARL
	{
	public:
		PEARL() : max_iteration_number(1000),
			inlier_outlier_threshold(2.0),
			spatial_coherence_weight(0.14),
			model_complexity_weight(0.1),
			alpha_expansion_engine(nullptr)
		{

		}

		bool run(const cv::Mat &data_,
			const _NeighborhoodGraph * neighborhood_graph_,
			_ModelEstimator &model_estimator_,
			std::vector<progx::Model<_ModelEstimator>> &models_);

	protected:
		GCoptimizationGeneralGraph *alpha_expansion_engine;
		double spatial_coherence_weight,
			model_complexity_weight,
			inlier_outlier_threshold;
		size_t point_number,
			max_iteration_number;

		bool labeling(const cv::Mat &data_,
			const std::vector<progx::Model<_ModelEstimator>> &models_,
			const _ModelEstimator &model_estimator_,
			const _NeighborhoodGraph &neighborhood_graph_,
			const bool changed_,
			double &energy_);

		const std::pair<std::vector<size_t>, size_t> getLabeling()
		{
			if (alpha_expansion_engine == nullptr)
				return std::make_pair<auto>(std::vector<size_t>(point_number, 0), 1);

			std::vector<size_t> labeling(point_number);
			size_t max_label = 0;
			for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			{
				const size_t &label = alpha_expansion_engine->whatLabel(point_idx);
				labeling[point_idx] = label;
				max_label = MAX(max_label, label);				
			}

			return std::make_pair<auto>(labeling, max_label + 1);
		}
	};

	template<class _NeighborhoodGraph,
		class _ModelEstimator>
		bool PEARL<_NeighborhoodGraph, _ModelEstimator>::run(
			const cv::Mat &data_,
			const _NeighborhoodGraph * neighborhood_graph_,
			_ModelEstimator &model_estimator_,
			std::vector<progx::Model<_ModelEstimator>> &models_)
	{
		point_number = data_.rows; // The number of points
		size_t iteration_number = 0, // The number of current iterations
			iteration_number_without_change; // The number of consecutive iterations when nothing has changed.
		double energy; // The energy of the alpha-expansion
		bool convergenve = false, // A flag to see if the algorithm has converges
			is_changed = false; // A flag to see if anything is changed in two consecutive iterations

		while (!convergenve &&
			iteration_number++ < max_iteration_number)
		{
			// Apply alpha-expansion to get the labeling which assigns each point to a model instance
			labeling(data_, // All data points
				model_estimator_, // The model estimator
				*neighborhood_graph_, // The neighborhood graph
				is_changed, // A flag to see if anything is changed in two consecutive iterations
				energy); // The energy of the alpha-expansion

			break;
		}
		return true;
	}

	template<class _NeighborhoodGraph,
		class _ModelEstimator>
		bool PEARL<_NeighborhoodGraph, _ModelEstimator>::labeling(
			const cv::Mat &data_,
			const std::vector<progx::Model<_ModelEstimator>> &models_,
			const _ModelEstimator &model_estimator_,
			const _NeighborhoodGraph &neighborhood_graph_,
			const bool changed_,
			double &energy_)
	{
		// Return if there are no model instances
		if (models_.size() == 0)
			return fales;

		// Set the previous labeling if nothing has changed
		std::vector<size_t> previous_labeling;
		if (!changed_ &&
			alpha_expansion_engine != nullptr)
		{
			previous_labeling.resize(point_number);
			for (size_t i = 0; i < point_number.size(); ++i)
				previous_labeling[i] = alpha_expansion_engine->whatLabel(i);
		}

		// Delete the alpha-expansion engine if it has been used.
		// The graph provided by the GCOptimization library is 
		// not reusable.
		if (alpha_expansion_engine != nullptr)
			delete alpha_expansion_engine;

		// Initializing the alpha-expansion engine with the given number of points and
		// with model number plus one labels. The plus labels is for the outlier class.
		alpha_expansion_engine =
			new GCoptimizationGeneralGraph(point_number, models_.size() + 1);

		// The object consisting of all information required for the energy calculations
		EnergyDataStructure<_ModelEstimator> information_object(
			data_, // The data points
			models_, // The model instances represented by the labels
			spatial_coherence_weight, // The weight of the spatial coherence term
			inlier_outlier_threshold); // The inlier-outlier threshold used when assigning points to model instances

		// Set the data cost functor to the alpha-expansion engine
		alpha_expansion_engine->setDataCost(&dataEnergyFunctor, // The data cost functor
			&information_object); // The object consisting of all information required for the energy calculations

		// Set the spatial cost functor to the alpha-expansion engine if needed
		if (spatial_coherence_weight > 0.0)
			alpha_expansion_engine->setSmoothCost(&spatialCoherenceEnergyFunctor, // The spatial cost functor
				&information_object); // The object consisting of all information required for the energy calculations

		// Set the model complexity weight to the alpha-expansion engine if needed
		if (model_complexity_weight > 0.0)
			alpha_expansion_engine->setLabelCost(model_complexity_weight);

		// Set neighbourhood of each point
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
			for (const size_t &neighbor_idx : neighborhood_graph_.getNeighbors(point_idx))
				if (point_idx != neighbor_idx)
					alpha_expansion_engine->setNeighbors(point_idx, neighbor_idx);
		
		// If nothing has changed since the previous labeling, use
		// the previous labels as initial values.
		if (!changed_ && 
			previous_labeling.size() > 0)
		{
			for (size_t point_idx = 0; i < point_number; ++point_idx)
				alpha_expansion_engine->setLabel(i, previous_labeling[point_idx]);
			previous_labeling.clear();
		}

		int iteration_number;
		energy = alpha_expansion_engine->expansion(iteration_number, 
			1000);
		
		return true;
	}
}
