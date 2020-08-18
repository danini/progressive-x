#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "progressivex_python.h"

namespace py = pybind11;

py::tuple find6DPoses(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  K_,
	double threshold,
	int max_model_number,
	double conf,
	double proposal_engine_conf,
	double spatial_coherence_weight,
	double neighborhood_ball_radius,
	double max_tanimoto_similarity,
	double scaling_from_millimeters,
	double min_triangle_area,
	double min_coverage,
	int max_iters,
	int min_point_number,
	bool use_prosac,
	size_t max_model_number_for_optimization,
	bool apply_numerical_optimization,
	bool log) 
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=3");
	}
	
	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2z2 should be the same size");
	}
	
	py::buffer_info buf1K = K_.request();
	size_t DIMK1 = buf1K.shape[0];
	size_t DIMK2 = buf1K.shape[1];

	if (DIMK1 != 3 || DIMK2 != 3) {
		throw std::invalid_argument("K should be an array with dims [3,3]");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

	double *ptr1K = (double *)buf1K.ptr;
	std::vector<double> K;
	K.assign(ptr1K, ptr1K + buf1K.size);

	std::vector<double> poses;
	std::vector<double> scores;
	std::vector<size_t> labeling(NUM_TENTS);

	int num_models = find6DPoses_(
		x1y1,
		x2y2z2,
		K,
		labeling,
		poses,
		scores,
		spatial_coherence_weight,
		threshold,
		conf,
		proposal_engine_conf,
		neighborhood_ball_radius,
		max_tanimoto_similarity,
		scaling_from_millimeters,
		min_coverage,
		min_triangle_area,
		max_iters,
		min_point_number,
		max_model_number,
		use_prosac,
		max_model_number_for_optimization,
		apply_numerical_optimization,
		log);

	py::array_t<double> scores_ = py::array_t<double>(num_models);
	py::buffer_info buf4 = scores_.request();
	double *ptr4 = (double *)buf4.ptr;
	for (size_t i = 0; i < num_models; i++)
		ptr4[i] = static_cast<double>(scores[i]);

	py::array_t<int> labeling_ = py::array_t<int>(NUM_TENTS);
	py::buffer_info buf3 = labeling_.request();
	int *ptr3 = (int *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = static_cast<int>(labeling[i]);
	
	py::array_t<double> poses_ = py::array_t<double>({ static_cast<size_t>(num_models) * 3, 4 });
	py::buffer_info buf2 = poses_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 12 * num_models; i++)
		ptr2[i] = poses[i];
	return py::make_tuple(poses_, labeling_, scores_);
}

PYBIND11_PLUGIN(pyprogressivex) {
                                                                             
    py::module m("pyprogressivex", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pyprogressivex
        .. autosummary::
           :toctree: _generate
           
           find6DPoses,

    )doc");
	
	m.def("find6DPoses", &find6DPoses, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2z2"),
		py::arg("K"),
		py::arg("threshold") = 4.0,
		py::arg("max_model_number") = -1,
		py::arg("conf") = 0.50,
		py::arg("proposal_engine_conf") = 1.00,
		py::arg("spatial_coherence_weight") = 0.1,
		py::arg("neighborhood_ball_radius") = 20.0,
		py::arg("max_tanimoto_similarity") = 0.9,
		py::arg("scaling_from_millimeters") = 0.1,
		py::arg("min_triangle_area") = 100,
		py::arg("min_coverage") = 0.5,
		py::arg("max_iters") = 400,
		py::arg("min_point_number") = 2 * 3,
		py::arg("use_prosac") = false,
		py::arg("max_model_number_for_optimization") = 3,
		py::arg("apply_numerical_optimization") = true,
		py::arg("log") = false);

  return m.ptr();
}
