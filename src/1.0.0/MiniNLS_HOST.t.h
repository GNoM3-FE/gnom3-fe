//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//

namespace LCM
{

//
// MiniSolver
//
template<
typename MIN, typename STEP, typename FN, typename EvalT, minitensor::Index N>
MiniSolver<MIN, STEP, FN, EvalT, N>::
MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      minitensor::Vector<typename EvalT::ScalarT, N> & soln)
{
  std::cerr << __PRETTY_FUNCTION__ << '\n';
  std::cerr << "ERROR: Instantiation of default MiniSolver class.\n";
  std::cerr << "This means a MiniSolver specialization is missing.\n";
  exit(1);
  return;
}

template<typename MIN, typename STEP, typename FN, minitensor::Index N>
MiniSolver<MIN, STEP, FN, Host_Traits::Residual, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    minitensor::Vector<Host_Traits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(step_method, function, soln);
  return;
}

//
//
//
template<typename T, typename S, minitensor::Index N>
void
computeFADInfo(
    minitensor::Vector<T, N> const & r,
    minitensor::Tensor<S, N> const & DrDx,
    minitensor::Vector<T, N> & x)
{
  // Check whether dealing with AD type.
  if (Sacado::IsADType<T>::value == false) return;

  //Deal with derivative information
  auto const
  dimension = r.get_dimension();

  assert(dimension > 0);

  auto const
  order = r[0].size();

  // No FAD info. Nothing to do.
  if (order == 0) return;

  // Extract sensitivities of r wrt p
  minitensor::Matrix<S, N>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  minitensor::Matrix<S, N>
  DxDp = minitensor::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }
}



} // namespace LCM
