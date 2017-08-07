//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//

#if !defined(MiniNonlinearSolver_h)
#define MiniNonlinearSolver_h

#include <type_traits>

#include "MiniTensor_Solvers.h"
#include "Host_Traits.hpp"

namespace LCM {

//
// Class for dealing with evaluation traits.
//
template<
typename MIN, typename STEP, typename FN, typename EvalT, minitensor::Index N>
struct MiniSolver
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      minitensor::Vector<typename EvalT::ScalarT, N> & soln);
};


template<typename MIN, typename STEP, typename FN, minitensor::Index N>
struct MiniSolver<MIN, STEP, FN, Host_Traits::Residual, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      minitensor::Vector<Host_Traits::Residual::ScalarT, N> & soln);
};


///
/// Deal with derivative information for all the mini solvers.
/// Call this when a converged solution is obtained on a system that is
/// typed on a FAD type.
/// Assuming that T is a FAD type and S is a simple type.
///
template<typename T, typename S, minitensor::Index N>
void
computeFADInfo(
    minitensor::Vector<T, N> const & r,
    minitensor::Tensor<S, N> const & DrDx,
    minitensor::Vector<T, N> & x);

///
/// Auxiliary functors that peel off derivative information from evaluation
/// types when not needed and keep it when needed. Used to convert types
/// within MiniSolver function class methods.
/// The type for N must be int to work with Sacado.
///
template<typename EvalT, typename T, int N>
struct peel
{
  using S = typename EvalT::ScalarT;

  // This ugly return type is to avoid matching Tensor types.
  // If it does not match then it just becomes T.
  using RET = typename
      minitensor::disable_if_c<minitensor::order_1234<T>::value, T>::type;

  RET
  operator()(S const & s)
  {
    T const
    t = s;

    return t;
  }
};

namespace {

using RE = Host_Traits::Residual;

template<int N>
using AD = minitensor::FAD<Real, N>;

} // anonymous namespace

template<int N>
struct peel<RE, Real, N>
{
  Real
  operator()(RE::ScalarT const & s)
  {
    Real const
    t = s;

    return t;
  }
};

template<int N>
struct peel<RE, AD<N>, N>
{
  Real
  operator()(typename RE::ScalarT const & s)
  {
    Real const
    t = s;

    return t;
  }
};

// M: number of derivatives
// N: vector/tensor dimension
template<typename EvalT, typename T, int M, int N>
struct peel_vector
{
  using S = typename EvalT::ScalarT;

  minitensor::Vector<T, N>
  operator()(minitensor::Vector<S, N> const & s)
  {
    minitensor::Index const
    dimension = s.get_dimension();

    minitensor::Vector<T, N>
    t(dimension);

    minitensor::Index const
    num_components = s.get_number_components();

    for (minitensor::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor
{
  using S = typename EvalT::ScalarT;

  minitensor::Tensor<T, N>
  operator()(minitensor::Tensor<S, N> const & s)
  {
    minitensor::Index const
    dimension = s.get_dimension();

    minitensor::Tensor<T, N>
    t(dimension);

    minitensor::Index const
    num_components = s.get_number_components();

    for (minitensor::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor3
{
  using S = typename EvalT::ScalarT;

  minitensor::Tensor3<T, N>
  operator()(minitensor::Tensor3<S, N> const & s)
  {
    minitensor::Index const
    dimension = s.get_dimension();

    minitensor::Tensor3<T, N>
    t(dimension);

    minitensor::Index const
    num_components = s.get_number_components();

    for (minitensor::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor4
{
  using S = typename EvalT::ScalarT;

  minitensor::Tensor4<T, N>
  operator()(minitensor::Tensor4<S, N> const & s)
  {
    minitensor::Index const
    dimension = s.get_dimension();

    minitensor::Tensor4<T, N>
    t(dimension);

    minitensor::Index const
    num_components = s.get_number_components();

    for (minitensor::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

} 

#include "MiniNLS_HOST.t.h"

#endif 
