//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid2_MiniTensor.h>
#include "Teuchos_TestForException.hpp"

#include "MiniSolver/MiniNLS_Alegra.h"

namespace FM
{

/******************************************************************************/
template<typename EvalT>
FerroicModel<EvalT>::FerroicModel(){}
/******************************************************************************/

/******************************************************************************/
template<typename EvalT>
void
FerroicModel<EvalT>::Parse(const Teuchos::ParameterList& p)
/******************************************************************************/
{

  // PARSE MATERIAL BASIS
  //
  Intrepid2::Tensor<RealType, FM::FM_3D>& basis = this->getBasis();
  basis.set_dimension(FM::FM_3D); basis.clear();
  if(p.isType<Teuchos::ParameterList>("Material Basis")){
    const Teuchos::ParameterList& pBasis = p.get<Teuchos::ParameterList>("Material Basis");
    FM::parseBasis(pBasis,basis);
  } else {
    basis(0,0) = 1.0; basis(1,1) = 1.0; basis(2,2) = 1.0;
  }


  // PARSE INITIAL BIN FRACTIONS
  //
  Teuchos::Array<RealType>&
    initialBinFractions = this->getInitialBinFractions();
  if(p.isType<Teuchos::Array<RealType>>("Bin Fractions") )
    initialBinFractions = p.get<Teuchos::Array<RealType>>("Bin Fractions");
  else
    initialBinFractions.resize(0);


  // PARSE PHASES
  //
  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >& crystalPhases = this->getCrystalPhases();
  int nphases = p.get<int>("Number of Phases");
  for(int i=0; i<nphases; i++){
    const Teuchos::ParameterList& pParam = p.get<Teuchos::ParameterList>(FM::strint("Phase",i+1));
    Intrepid2::Tensor4<RealType, FM::FM_3D> C;   FM::parseTensor4(pParam, C  );
    Intrepid2::Tensor3<RealType, FM::FM_3D> ep;   FM::parseTensor3(pParam, ep  );
    Intrepid2::Tensor <RealType, FM::FM_3D> k; FM::parseTensor (pParam, k);
    crystalPhases.push_back(Teuchos::rcp(new FM::CrystalPhase(basis, C, ep, k)));
  }

  // PARSE VARIANTS
  //
  Teuchos::Array<FM::CrystalVariant>& crystalVariants = this->getCrystalVariants();
  if(initialBinFractions.size() > 0){
    const Teuchos::ParameterList& vParams = p.get<Teuchos::ParameterList>("Variants");
    int nvars = vParams.get<int>("Number of Variants");
    crystalVariants.resize(nvars);
    TEUCHOS_TEST_FOR_EXCEPTION(initialBinFractions.size() != nvars, std::invalid_argument,
       ">>> ERROR (FerroicModel): 'Number of Variants' must equal length of 'Bin Fractions' array");
    for(int i=0; i<nvars; i++){
      const Teuchos::ParameterList& vParam = vParams.get<Teuchos::ParameterList>(FM::strint("Variant",i+1));
      crystalVariants[i] = parseCrystalVariant(crystalPhases, vParam);
    }
  } else {
    // no variants specified.  Create single dummy variant.
    initialBinFractions.resize(1);
    initialBinFractions[0] = 1.0;
  }

  // PARSE CRITICAL ENERGIES
  //
  int nVariants = crystalVariants.size();
  Teuchos::Array<RealType>& tBarrier = this->getTransitionBarrier();
  tBarrier.resize(nVariants*nVariants);
  if(p.isType<Teuchos::ParameterList>("Critical Values")){
    const Teuchos::ParameterList& cParams = p.get<Teuchos::ParameterList>("Critical Values");
    int transitionIndex = 0;
    for(int i=0; i<nVariants; i++){
      Teuchos::Array<RealType> array = cParams.get<Teuchos::Array<RealType>>(FM::strint("Variant",i+1));
      TEUCHOS_TEST_FOR_EXCEPTION(array.size()!=nVariants, std::invalid_argument,
         ">>> ERROR (FerroicModel): List of critical values for variant " << i+1 << " is wrong length");
      for(int j=0; j<nVariants; j++){
        tBarrier[transitionIndex] = array[j];
        transitionIndex++;
      }
    }
  }
}

/******************************************************************************/
template<typename EvalT>
void
FerroicModel<EvalT>::PostParseInitialize()
/******************************************************************************/
{

  // create transitions
  //
  int nVariants = crystalVariants.size();
  transitions.resize(nVariants*nVariants);
  int transIndex = 0;
  for(int I=0; I<nVariants; I++)
    for(int J=0; J<nVariants; J++){
      FM::Transition& t = transitions[transIndex];
      CrystalVariant& fromVariant = crystalVariants[I];
      CrystalVariant& toVariant = crystalVariants[J];
      t.transStrain = toVariant.spontStrain - fromVariant.spontStrain;
      t.transEDisp  = toVariant.spontEDisp  - fromVariant.spontEDisp;
      transIndex++;
    }

  // create/initialize transition matrix
  //
  int nTransitions = transitions.size();
  aMatrix.resize(nVariants, nTransitions);
  for(int I=0; I<nVariants; I++){
    for(int J=0; J<nVariants; J++){
      aMatrix(I,nVariants*I+J) = -1.0;
      aMatrix(J,nVariants*I+J) = 1.0;
    }
    aMatrix(I,nVariants*I+I) = 0.0;
  }
}

/******************************************************************************/
template<typename EvalT>
void FerroicModel<EvalT>::
computeState(
      const Intrepid2::Tensor<ScalarT, FM::FM_3D>& x,
      const Intrepid2::Vector<ScalarT, FM::FM_3D>& E,
      const Teuchos::Array<RealType>& oldfractions,
            Intrepid2::Tensor<ScalarT, FM::FM_3D>& X,
            Intrepid2::Vector<ScalarT, FM::FM_3D>& D,
            Teuchos::Array<ScalarT>& newfractions)
/******************************************************************************/
{

  // create non-linear system
  //
  using NLS = FM::DomainSwitching<EvalT>;
  NLS domainSwitching(crystalVariants, transitions, tBarriers,
                      oldfractions, aMatrix, x, E, /* dt= */ 1.0);

  // solution variable
  //
  Intrepid2::Vector<ScalarT,FM::MAX_TRNS> xi;
  // create solution vector with initial guess
  int ntrans = domainSwitching.getNumStates();
  xi.set_dimension(ntrans);
  xi.clear();

  // solve for xi
  //

  // JR todo: don't hardwire:
  m_integrationType = FM::IntegrationType::EXPLICIT;
  // m_explicitMethod = FM::ExplicitMethod::SCALED_DESCENT;
  m_explicitMethod = FM::ExplicitMethod::DESCENT_NORM;

  switch (m_integrationType){

    default:
    break;

    case FM::IntegrationType::EXPLICIT:
    {

      switch (m_explicitMethod){

        default:
        break;

        case FM::ExplicitMethod::SCALED_DESCENT:
        {
          FM::ScaledDescent(domainSwitching, xi);
          break;
        }
        case FM::ExplicitMethod::DESCENT_NORM:
        {
          FM::DescentNorm(domainSwitching, xi);
          break;
        }
      }
      break;
    }

    case FM::IntegrationType::IMPLICIT:
    {

      // create minimizer
      using ValueT = typename Sacado::ValueType<ScalarT>::type;
      using MIN = Intrepid2::Minimizer<ValueT, MAX_TRNS>;
      MIN minimizer;

      // create stepper
      using STEP = Intrepid2::StepBase<NLS, ValueT, MAX_TRNS>;
      std::unique_ptr<STEP>
        pstep = Intrepid2::stepFactory<NLS, ValueT, MAX_TRNS>(m_step_type);
      STEP &step = *pstep;

      for(int itrans=0; itrans<ntrans; itrans++)
        xi(itrans) = Sacado::ScalarValue<ScalarT>::eval(0.0);

      // solve
      LCM::MiniSolver<MIN, STEP, NLS, EvalT, MAX_TRNS>
        mini_solver(minimizer, step, domainSwitching, xi);

      break;
    }
  }

  // update based on new xi values
  //
  const Teuchos::Array<int>& transitionMap = domainSwitching.getTransitionMap();
  FM::computeBinFractions(xi, newfractions, oldfractions, transitionMap, aMatrix);

  Intrepid2::Tensor<ScalarT, FM::FM_3D> linear_x;
  linear_x.clear();

  Intrepid2::Vector<ScalarT, FM::FM_3D> linear_D, E_relaxed;
  linear_D.clear();
  E_relaxed.clear();


  //FM::computeInitialState(newfractions, crystalVariants,
  //                        x, X, linear_x,
  //                        E, D, linear_D);

  FM::computeInitialState(oldfractions, crystalVariants,
                          x, X, linear_x,
                          E, D, linear_D);

  FM::computeRelaxedState(newfractions, crystalVariants,
                          x, X, linear_x,
                          E_relaxed, D, linear_D);




}

}
