
#include "deform_conv2d.h"
#include "modulated_deform_conv2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_conv2d_forward", &deform_conv2d_forward, "deform_conv2d_forward");
  m.def("deform_conv2d_backward", &deform_conv2d_backward, "deform_conv2d_backward");
  m.def("modulated_deform_conv2d_forward", &modulated_deform_conv2d_forward, "modulated_deform_conv2d_forward");
  m.def("modulated_deform_conv2d_backward", &modulated_deform_conv2d_backward, "modulated_deform_conv2d_backward");
}
