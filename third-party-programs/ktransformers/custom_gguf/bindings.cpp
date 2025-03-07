#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// Include your header file
#include "dequant.dp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize_iq4", &dequantize_iq4, "Dequantize IQ4 function");
    // Add other dequantize_xxxx functions here
}