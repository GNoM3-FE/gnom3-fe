//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

namespace LCM
{

//
// MiniSolverr
//
template<
typename MIN, typename STEP, typename FN, typename EvalT, Intrepid2::Index N>
MiniSolver<MIN, STEP, FN, EvalT, N>::
MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<typename EvalT::ScalarT, N> & soln)
{
  std::cerr << __PRETTY_FUNCTION__ << '\n';
  std::cerr << "ERROR: Instantiation of default MiniSolver class.\n";
  std::cerr << "This means a MiniSolver specialization is missing.\n";
  exit(1);
  return;
}

template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
MiniSolver<MIN, STEP, FN, AlegraTraits::Residual, N>::
MiniSolver(
    MIN & minimizer,
    STEP & step_method,
    FN & function,
    Intrepid2::Vector<AlegraTraits::Residual::ScalarT, N> & soln)
{
  minimizer.solve(step_method, function, soln);
  return;
}

//
//
//
template<typename T, typename S, Intrepid2::Index N>
void
computeFADInfo(
    Intrepid2::Vector<T, N> const & r,
    Intrepid2::Tensor<S, N> const & DrDx,
    Intrepid2::Vector<T, N> & x)
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
  Intrepid2::Matrix<S, N>
  DrDp(dimension, order);

  for (auto i = 0; i < dimension; ++i) {
    for (auto j = 0; j < order; ++j) {
      DrDp(i, j) = r(i).dx(j);
    }
  }

  // Solve for all DxDp
  Intrepid2::Matrix<S, N>
  DxDp = Intrepid2::solve(DrDx, DrDp);

  // Pack into x.
  for (auto i = 0; i < dimension; ++i) {
    x(i).resize(order);
    for (auto j = 0; j < order; ++j) {
      x(i).fastAccessDx(j) = -DxDp(i, j);
    }
  }
}

#ifdef ALBANY_ENSEMBLE
#endif // ALBANY_ENSEMBLE

} // namespace LCM
