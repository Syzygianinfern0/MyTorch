#include "deform_modules.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("deform_conv_forward", &deform_conv_forward_cuda, "deform_conv_forward_cuda");
  m.def("deform_conv_backward_input", &deform_conv_backward_input_cuda, "deform_conv_backward_input_cuda");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters_cuda, "deform_conv_backward_parameters_cuda");
  m.def("deform_unfold_forward", &deform_unfold_forward_cuda, "deform unfold forward_cuda");
  m.def("deform_unfold_backward_input", &deform_unfold_backward_input_cuda, "deform_unfold_backward_input_cuda");
}
