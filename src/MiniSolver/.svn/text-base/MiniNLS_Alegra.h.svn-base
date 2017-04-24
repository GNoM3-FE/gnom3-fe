#if !defined(MiniNonlinearSolver_h)
#define MiniNonlinearSolver_h

#include <type_traits>

#include "Intrepid2_MiniTensor_Solvers.h"
#include "AlegraTraits.hpp"

namespace LCM {

//
// Class for dealing with evaluation traits.
//
template<
typename MIN, typename STEP, typename FN, typename EvalT, Intrepid2::Index N>
struct MiniSolver
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<typename EvalT::ScalarT, N> & soln);
};


template<typename MIN, typename STEP, typename FN, Intrepid2::Index N>
struct MiniSolver<MIN, STEP, FN, AlegraTraits::Residual, N>
{
  MiniSolver(
      MIN & minimizer,
      STEP & step_method,
      FN & function,
      Intrepid2::Vector<AlegraTraits::Residual::ScalarT, N> & soln);
};


///
/// Deal with derivative information for all the mini solvers.
/// Call this when a converged solution is obtained on a system that is
/// typed on a FAD type.
/// Assuming that T is a FAD type and S is a simple type.
///
template<typename T, typename S, Intrepid2::Index N>
void
computeFADInfo(
    Intrepid2::Vector<T, N> const & r,
    Intrepid2::Tensor<S, N> const & DrDx,
    Intrepid2::Vector<T, N> & x);

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
      Intrepid2::disable_if_c<Intrepid2::order_1234<T>::value, T>::type;

  RET
  operator()(S const & s)
  {
    T const
    t = s;

    return t;
  }
};

namespace {

using RE = AlegraTraits::Residual;

template<int N>
using AD = Intrepid2::FAD<Real, N>;

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

  Intrepid2::Vector<T, N>
  operator()(Intrepid2::Vector<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Vector<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor<T, N>
  operator()(Intrepid2::Tensor<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor3
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor3<T, N>
  operator()(Intrepid2::Tensor3<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor3<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

template<typename EvalT, typename T, int M, int N>
struct peel_tensor4
{
  using S = typename EvalT::ScalarT;

  Intrepid2::Tensor4<T, N>
  operator()(Intrepid2::Tensor4<S, N> const & s)
  {
    Intrepid2::Index const
    dimension = s.get_dimension();

    Intrepid2::Tensor4<T, N>
    t(dimension);

    Intrepid2::Index const
    num_components = s.get_number_components();

    for (Intrepid2::Index i = 0; i < num_components; ++i) {
      t[i] = peel<EvalT, T, M>()(s[i]);
    }

    return t;
  }
};

} 

#include "MiniNLS_Alegra.t.h"

#endif 
