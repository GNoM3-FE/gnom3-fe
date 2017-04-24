#if !defined(FerroicCore_hpp)
#define FerroicCore_hpp

#include <Intrepid_FieldContainer.hpp>
#include <Intrepid2_MiniTensor.h>
#include "MiniSolver/MiniNLS_Alegra.h"

namespace FM
{

static constexpr Intrepid2::Index FM_3D = 3;
static constexpr Intrepid2::Index MAX_VRNT = 11;
static constexpr Intrepid2::Index MAX_TRNS = MAX_VRNT*MAX_VRNT;


enum class IntegrationType
{
  UNDEFINED = 0, EXPLICIT = 1, IMPLICIT = 2
};

enum class ExplicitMethod
{
  UNDEFINED = 0, SCALED_DESCENT = 1, DESCENT_NORM = 2
};


/******************************************************************************/
// Data structures
/******************************************************************************/

/******************************************************************************/
struct CrystalPhase
/******************************************************************************/
{
  CrystalPhase(Intrepid2::Tensor <RealType, FM_3D>& basis,
               Intrepid2::Tensor4<RealType, FM_3D>& C_matBasis,
               Intrepid2::Tensor3<RealType, FM_3D>& ep_matBasis,
               Intrepid2::Tensor <RealType, FM_3D>& k_matBasis);

  Intrepid2::Tensor4<RealType, FM_3D> C;
  Intrepid2::Tensor3<RealType, FM_3D> ep;
  Intrepid2::Tensor <RealType, FM_3D> k;
  Intrepid2::Tensor <RealType, FM_3D> basis;
};


/******************************************************************************/
struct CrystalVariant
/******************************************************************************/
{
  Intrepid2::Tensor4<RealType, FM_3D> C;
  Intrepid2::Tensor3<RealType, FM_3D> ep;
  Intrepid2::Tensor<RealType, FM_3D> k;
  Intrepid2::Tensor<RealType, FM_3D> basis;
  Intrepid2::Tensor<RealType, FM_3D> spontStrain;
  Intrepid2::Vector<RealType, FM_3D> spontEDisp;
};

/******************************************************************************/
struct Transition
/******************************************************************************/
{
  Intrepid2::Tensor<RealType, FM_3D> transStrain;
  Intrepid2::Vector<RealType, FM_3D> transEDisp;
};




/******************************************************************************/
// Service functions:
/******************************************************************************/

template<typename DataT>
void
changeBasis(      Intrepid2::Tensor4<DataT, FM_3D>& inMatlBasis,
            const Intrepid2::Tensor4<DataT, FM_3D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      Intrepid2::Tensor3<DataT, FM_3D>& inMatlBasis,
            const Intrepid2::Tensor3<DataT, FM_3D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      Intrepid2::Tensor <DataT, FM_3D>& inMatlBasis,
            const Intrepid2::Tensor <DataT, FM_3D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      Intrepid2::Vector <DataT, FM_3D>& inMatlBasis,
            const Intrepid2::Vector <DataT, FM_3D>& inGlblBasis,
            const Intrepid2::Tensor <DataT, FM_3D>& Basis);


template<typename NLS, typename DataT>
void
DescentNorm(NLS & nls, Intrepid2::Vector<DataT, MAX_TRNS> & xi);

template<typename NLS, typename DataT>
void
ScaledDescent(NLS & nls, Intrepid2::Vector<DataT, MAX_TRNS> & xi);

template<typename DataT, typename ArgT>
void
computeBinFractions(
    Intrepid2::Vector<ArgT, FM::MAX_TRNS> const & xi,
    Teuchos::Array<ArgT>                        & newFractions,
    Teuchos::Array<DataT>                 const & oldFractions,
    Teuchos::Array<int>                   const & transitionMap,
    Intrepid::FieldContainer<DataT>       const & aMatrix);


template<typename ArgT>
void
computeInitialState(
    Teuchos::Array<RealType>            const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::FM_3D> const & x,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & X,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::FM_3D> const & E,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & D,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & linear_D);


template<typename ArgT>
void
computeRelaxedState(
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::FM_3D> const & x,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & X,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & E,
    Intrepid2::Vector<ArgT,FM::FM_3D> const & D,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & linear_D);


template<typename DataT, typename ArgT>
void
computeResidual(
    Intrepid2::Vector<ArgT, FM::MAX_TRNS>       & residual,
    Teuchos::Array<ArgT>                  const & fractions,
    Teuchos::Array<int>                   const & transitionMap,
    Teuchos::Array<FM::Transition>        const & transitions,
    Teuchos::Array<FM::CrystalVariant>    const & crystalVariants,
    Teuchos::Array<DataT>                 const & tBarrier,
    Intrepid::FieldContainer<DataT>       const & aMatrix,
    Intrepid2::Tensor<ArgT,FM::FM_3D>   const & X,
    Intrepid2::Tensor<ArgT,FM::FM_3D>   const & linear_x,
    Intrepid2::Vector<ArgT,FM::FM_3D>   const & E,
    Intrepid2::Vector<ArgT,FM::FM_3D>   const & linear_D);


template<typename ArgT>
void
computePermittivity(
    Intrepid2::Tensor<ArgT,FM::FM_3D>         & eps,
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants);

template<typename ArgT>
void
computePolarization(
    Intrepid2::Vector<ArgT,FM::FM_3D>         & pol,
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants);


template<typename ArgT>
void
computeMaxCD(
        Teuchos::Array<ArgT>                    & max_CD,
        Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >  const & crystalPhases);



/******************************************************************************/
// Host-independent Models
/******************************************************************************/


//
//! Nonlinear Solver (NLS) class for the domain switching / phase transition model.
//  Unknowns: transition rates
//
template<typename EvalT>
class DomainSwitching:
    public Intrepid2::Function_Base<
    DomainSwitching<EvalT>, typename EvalT::ScalarT>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  DomainSwitching(
      Teuchos::Array<FM::CrystalVariant> const & crystalVariants,
      Teuchos::Array<FM::Transition>     const & transitions,
      Teuchos::Array<RealType>           const & transBarriers,
      Teuchos::Array<RealType>           const & binFractions,
      Intrepid::FieldContainer<RealType> const & aMatrix,
      Intrepid2::Tensor<ArgT,FM_3D>    const & x,
      Intrepid2::Vector<ArgT,FM_3D>    const & E,
      RealType dt);

  static constexpr char const * const NAME =
      "Domain Switching Nonlinear System";

  //! Default implementation of value function.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  T
  value(Intrepid2::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the
  // transition rate at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  gradient(Intrepid2::Vector<T, N> const & x) const;

  //! Xi function; returns the conditioned residual vector as a function of the
  // transition rate at step N+1.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  updateXi(Intrepid2::Vector<T, N> const & x);

  //! The gradient is conditioned such that it is multiplied by VFrac_from_old
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Vector<T, N>
  conditionedGrad(Intrepid2::Vector<T, N> const & x_new, Intrepid2::Vector<T, N> const & x_old);


  //! Default implementation of hessian function.
  template<typename T, Intrepid2::Index N = Intrepid2::DYNAMIC>
  Intrepid2::Tensor<T, N>
  hessian(Intrepid2::Vector<T, N> const & x);

  int getNumStates(){return m_numActiveTransitions;}

  const Teuchos::Array<int>& getTransitionMap(){ return m_transitionMap; }

private:

  Teuchos::Array<FM::CrystalVariant>  const & m_crystalVariants;
  Teuchos::Array<FM::Transition>      const & m_transitions;
  Teuchos::Array<RealType>            const & m_transBarriers;
  Teuchos::Array<RealType>            const & m_binFractions;
  Intrepid::FieldContainer<RealType>  const & m_aMatrix;
  Intrepid2::Tensor<ArgT,FM_3D>     const & m_x;
  Intrepid2::Vector<ArgT,FM_3D>             m_D;
  RealType m_dt;
  int m_numActiveTransitions;

  // JR todo:  put this in FerroicModel and pass in reference
  Teuchos::Array<int> m_transitionMap;
};

std::string strint(std::string s, int i, char delim = ' ');

} // namespace FM

#include "FerroicCore_Def.hpp"

#endif
