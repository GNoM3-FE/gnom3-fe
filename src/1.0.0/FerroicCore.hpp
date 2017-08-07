//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//


#if !defined(FerroicCore_hpp)
#define FerroicCore_hpp

#include "Intrepid_FieldContainer.hpp"
#include "MiniTensor.h"

// #include <MiniNonlinearSolver.h>         //FOR ALBANY
#include "MiniNLS_HOST.h"      //FOR ALEGRA

namespace FM
{
//THESE VALUES SHOULD NOT BE STATIC (MAX_VRNT, MAX_GRNS)
static constexpr minitensor::Index FM_3D = 3;
static constexpr minitensor::Index FM_SCAL = 1;  //scalars
static constexpr minitensor::Index FM_ARB = 100;  //arbitrary length

static constexpr minitensor::Index MAX_VRNT = 11;
static constexpr minitensor::Index MAX_TRNS = MAX_VRNT*MAX_VRNT;
static constexpr minitensor::Index MAX_AMTL = MAX_VRNT*MAX_TRNS;  //AMATRIX DIMENSION

static constexpr minitensor::Index MAX_GRNS = 32;  //max grains
static constexpr minitensor::Index MAX_GVTS = MAX_GRNS*MAX_VRNT;  //
static constexpr minitensor::Index MAX_GTRN = MAX_GRNS*MAX_TRNS;  //

static constexpr minitensor::Index MAX_PHAS = MAX_VRNT;


enum class IntegrationType
{
  UNDEFINED = 0, EXPLICIT = 1, IMPLICIT = 2
};

enum class ExplicitMethod
{
  UNDEFINED = 0, SCALED_DESCENT = 1, DESCENT_NORM = 2, EXPLICIT_SMOOTH = 3
};


/******************************************************************************/
// Data structures
/******************************************************************************/

/******************************************************************************/
struct CrystalPhase
/******************************************************************************/
{
  CrystalPhase(minitensor::Tensor <RealType, FM_3D>& R_sym, //change to R_phase usually no rotation
               minitensor::Tensor4<RealType, FM_3D>& C_sym,
               minitensor::Tensor3<RealType, FM_3D>& ep_sym,
               minitensor::Tensor <RealType, FM_3D>& k_sym);

  minitensor::Tensor4<RealType, FM_3D> C;
  minitensor::Tensor3<RealType, FM_3D> ep;
  minitensor::Tensor <RealType, FM_3D> k;
  minitensor::Tensor <RealType, FM_3D> R_phase_sym;
};


/******************************************************************************/
struct CrystalVariant
/******************************************************************************/
{
  minitensor::Tensor4<RealType, FM_3D> C;
  minitensor::Tensor3<RealType, FM_3D> ep;
  minitensor::Tensor<RealType, FM_3D> k;
  minitensor::Tensor<RealType, FM_3D> R_variant_phase; //change to R_phase_variant 
  minitensor::Tensor<RealType, FM_3D> spontStrain;
  minitensor::Vector<RealType, FM_3D> spontEDisp;
  int                                phaseIndex;
};

/******************************************************************************/
struct Transition
/******************************************************************************/
{
  minitensor::Tensor<RealType, FM_3D> transStrain;
  minitensor::Vector<RealType, FM_3D> transEDisp;
};




/******************************************************************************/
struct CrystalGrain
/******************************************************************************/
{
  minitensor::Tensor<RealType, FM::FM_3D>      R_grain_variant;
  Teuchos::Array<FM::CrystalVariant>          crystalVariants;
  Teuchos::Array<FM::Transition>              transitions;
  minitensor::Vector<RealType, FM::MAX_AMTL>   aMatrix;
  //Teuchos::Array<RealType>                  initialBinFractions;
  //Teuchos::Array<RealType>                  tBarriers;
  //Teuchos::Array<RealType>                  dBarriers;
  //Teuchos::Array<RealType>                  polyN;
  
  //Intrepid::FieldContainer<RealType>        aMatrix;
};




/******************************************************************************/
// Service functions:
/******************************************************************************/

template<typename DataT>
void
changeBasis(      minitensor::Tensor4<DataT, FM_3D>& inMatlBasis,
            const minitensor::Tensor4<DataT, FM_3D>& inGlblBasis,
            const minitensor::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      minitensor::Tensor3<DataT, FM_3D>& inMatlBasis,
            const minitensor::Tensor3<DataT, FM_3D>& inGlblBasis,
            const minitensor::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      minitensor::Tensor <DataT, FM_3D>& inMatlBasis,
            const minitensor::Tensor <DataT, FM_3D>& inGlblBasis,
            const minitensor::Tensor <DataT, FM_3D>& Basis);

template<typename DataT>
void
changeBasis(      minitensor::Vector <DataT, FM_3D>& inMatlBasis,
            const minitensor::Vector <DataT, FM_3D>& inGlblBasis,
            const minitensor::Tensor <DataT, FM_3D>& Basis);

template<typename NLS, typename DataT>
void
Explicit_Smooth(NLS & nls, minitensor::Vector<DataT, MAX_GTRN> & xi);


template<typename NLS, typename DataT>
void
Explicit_Smooth_Poling(NLS & nls, minitensor::Vector<DataT, MAX_GTRN> & xi);


template<typename NLS, typename DataT>
void
DescentNorm(NLS & nls, minitensor::Vector<DataT, MAX_TRNS> & xi);

template<typename NLS, typename DataT>
void
ScaledDescent(NLS & nls, minitensor::Vector<DataT, MAX_TRNS> & xi);

template<typename ArgT>
void
computeBinFractions(
    minitensor::Vector<ArgT, FM::MAX_TRNS>     const & xi,
    Teuchos::Array<ArgT>                            & newFractions,
    Teuchos::Array<ArgT>                      const & oldFractions,
    minitensor::Vector<RealType, FM::MAX_AMTL> const & aMatrix,
    Teuchos::Array<int>                       const & nVals);


template<typename ArgT>
void
computeInitialState(
    Teuchos::Array<RealType>          const & fractions,
    Teuchos::Array<FM::CrystalGrain>  const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D> const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>       & X,
    minitensor::Tensor<ArgT,FM::FM_3D>       & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D> const & E,
    minitensor::Vector<ArgT,FM::FM_3D>       & D,
    minitensor::Vector<ArgT,FM::FM_3D>       & linear_D,
    Teuchos::Array<int>               const & nVals);



template<typename ArgT>
void
computeFinalState(
    Teuchos::Array<ArgT>              const & fractions,
    Teuchos::Array<FM::CrystalGrain>  const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D> const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>       & X,
    minitensor::Tensor<ArgT,FM::FM_3D>       & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D> const & E,
    minitensor::Vector<ArgT,FM::FM_3D>       & D,
    minitensor::Vector<ArgT,FM::FM_3D>       & linear_D,
    Teuchos::Array<int>               const & nVals);


template<typename ArgT>
void
computeRelaxedState(
    Teuchos::Array<ArgT>              const & fractions,
    Teuchos::Array<FM::CrystalGrain>  const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D> const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>       & X,
    minitensor::Tensor<ArgT,FM::FM_3D>       & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D>       & E,
    minitensor::Vector<ArgT,FM::FM_3D> const & D,
    minitensor::Vector<ArgT,FM::FM_3D>       & linear_D,
    Teuchos::Array<int>               const & nVals);


template<typename DataT, typename ArgT>
void
computeResidual(
    minitensor::Vector<ArgT, FM::MAX_GTRN>         & residual,
    Teuchos::Array<FM::CrystalGrain>        const & crystalGrains,
    Teuchos::Array<DataT>                   const & tBarrier,
    minitensor::Tensor<ArgT,FM::FM_3D>       const & X,
    minitensor::Vector<ArgT,FM::FM_3D>       const & E,
    Teuchos::Array<int>                     const & nVals);
    



template<typename ArgT>
void
computeToAlegra(
    minitensor::Vector<ArgT,FM::FM_3D>         & nd_pol,         //non dielectric polarization
    minitensor::Tensor<ArgT,FM::FM_3D>         & eps,            //permittivity tensor
    minitensor::Vector<ArgT,FM::MAX_PHAS>      & vphases,        //volfrac of phases
    minitensor::Vector<ArgT,FM::FM_SCAL>       & pressure,       //pressure
    minitensor::Vector<ArgT,FM::FM_SCAL>       & PVFactor,       //pressure
    Teuchos::Array<RealType>            const & oldfractions,
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalGrain>    const & crystalGrains,
    minitensor::Vector<ArgT,FM::FM_3D>   const & E,
    minitensor::Tensor<ArgT,FM::FM_3D>   const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>   const & X,
    Teuchos::Array<int>                 const & nVals);





template<typename ArgT>
void
computeMaxCD(
        Teuchos::Array<ArgT>                                    & max_CD,
        Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >   const & crystalPhases,
        Teuchos::Array<int>                               const & nVals);



template<typename ArgT>
void
ES_computePhiNew(
    minitensor::Vector<ArgT, FM::MAX_TRNS>            & phi_new,
    minitensor::Vector<ArgT, FM::MAX_TRNS>      const & grad,
    Teuchos::Array<RealType>                   const & deltaG,
    Teuchos::Array<RealType>                   const & npower,
    int                                        const nTrans);



template<typename ArgT>
void
ES_computePhiOld(
    minitensor::Vector<ArgT, FM::MAX_GTRN>            & phi_old,
    minitensor::Vector<ArgT, FM::MAX_GVTS>      const & oldVolFrac,
    int                                        const nvars,
    int                                        const ngrains);
    
template<typename ArgT>
void
ES_solveCondLE(
    minitensor::Vector<ArgT, FM::MAX_TRNS>      const & A,
    minitensor::Vector<ArgT, FM::MAX_VRNT>            & X,
    minitensor::Vector<ArgT, FM::MAX_VRNT>      const & B,
    int                                        const nVars,
    int                                        const i,
    minitensor::Vector<ArgT, FM::MAX_TRNS>            & Aeye);
    
    








/******************************************************************************/
// Host-independent Models
/******************************************************************************/


//
//! Nonlinear Solver (NLS) class for the domain switching / phase transition model.
//  Unknowns: transition rates
//
template<typename EvalT>
class DomainSwitching:
    public minitensor::Function_Base<
    DomainSwitching<EvalT>, typename EvalT::ScalarT, FM::MAX_GRNS>
{
  using ArgT = typename EvalT::ScalarT;

public:

  //! Constructor.
  DomainSwitching(
      Teuchos::Array<FM::CrystalGrain>   const & crystalGrains,
      Teuchos::Array<RealType>           const & transBarriers,
      Teuchos::Array<RealType>           const & deltaBarriers,
      Teuchos::Array<RealType>           const & polynomialN,
      Teuchos::Array<RealType>           const & binFractions,
      minitensor::Tensor<ArgT,FM_3D>      const & x,
      minitensor::Vector<ArgT,FM_3D>      const & E,
      RealType                                   dt,
      Teuchos::Array<int>                const & nVals);

  static constexpr char const * const NAME =
      "Domain Switching Nonlinear System";

  //! Default implementation of value function.
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  T
  value(minitensor::Vector<T, N> const & x);

  //! Gradient function; returns the residual vector as a function of the
  // transition rate at step N+1.
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Vector<T, N>
  gradient(minitensor::Vector<T, N> const & x) const;
  

  //! Gradient function; returns the residual vector as a function of the
  // transition rate at step N+1.
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Vector<T, N>
  gradientStiff(minitensor::Vector<T, N> const & x) const;


  //! Default implementation of hessian function.
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Tensor<T, N>
  hessian(minitensor::Vector<T, N> const & x);
  
  //! Returns the new Volume Fractions
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Vector<T, FM::MAX_GVTS>
  computeVolFrac(minitensor::Vector<T, N> const & xi);
  
  
  //! Returns newPhi (Target Phi)
  template<typename T, minitensor::Index N = minitensor::DYNAMIC>
  minitensor::Vector<T, N>
  phiNew(minitensor::Vector<T, N> const & xi);
  
  
  
  
  const Teuchos::Array<int>& get_nVals() {return m_nVals; }
  
  
  //int getNumStates(){return m_numActiveTransitions;}

  //const Teuchos::Array<int>& getTransitionMap(){ return m_transitionMap; }

private:

  Teuchos::Array<FM::CrystalGrain>    const & m_crystalGrains;
  Teuchos::Array<RealType>            const & m_transBarriers;
  Teuchos::Array<RealType>            const & m_deltaBarriers;
  Teuchos::Array<RealType>            const & m_polynomialN;
  Teuchos::Array<RealType>            const & m_binFractions;
  minitensor::Tensor<ArgT,FM_3D>       const & m_x;
  minitensor::Vector<ArgT,FM_3D>       const & m_E;
  minitensor::Vector<ArgT,FM_3D>               m_D;
  RealType                                    m_dt;
  Teuchos::Array<int>                 const & m_nVals;
  //int m_numActiveTransitions;

  // JR todo:  put this in FerroicModel and pass in reference
  //Teuchos::Array<int> m_transitionMap;
};

std::string strint(std::string s, int i, char delim = ' ');

} // namespace FM

#include "FerroicCore_Def.hpp"

#endif
