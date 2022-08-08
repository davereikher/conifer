#include <vector>
#include <algorithm>
#include "firmware/my_prj.h"
#include "firmware/parameters.h"
#include "firmware/BDT.h"

std::vector<double> decision_function(std::vector<double> x){
  /* Do the prediction with data in/out as double, cast to input_t before prediction */
  std::vector<input_t> xtv;
  std::transform(x.begin(), x.end(), std::back_inserter(xtv),
                  [](double xi) -> input_t { return (input_t) xi; });
  input_arr_t xt;
  std::copy(xtv.begin(), xtv.end(), xt);

  final_score_t yt[BDT::fn_classes(n_classes)];
  tree_score_t tree_scores[BDT::fn_classes(n_classes) * n_trees];

  bdt.decision_function(xt, yt, tree_scores);
  std::vector<final_score_t> ytv(yt, yt + BDT::fn_classes(n_classes));
  std::vector<double> yd;
  std::transform(ytv.begin(), ytv.end(), std::back_inserter(yd),
                  [](final_score_t yi) -> double { return (double) yi; });
  return yd;  
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(conifer_bridge, m) {
  m.def("decision_function", &decision_function);
}
