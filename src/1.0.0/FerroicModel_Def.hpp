//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//




#include "MiniTensor.h"
#include "Teuchos_TestForException.hpp"


#include "MiniNLS_HOST.h"       //ALEGRA


//#include "Phalanx_DataLayout.hpp"               //ALBANY
//#include "Albany_Utils.hpp"
//
//#include <MiniNonlinearSolver.h>

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

  int ngrains;
  int nvars;
  int nphases;
  
  

  // PARSE MATERIAL BASIS
  //
  minitensor::Tensor<RealType, FM::FM_3D>& R_mat_grain = this->getBasis();
  R_mat_grain.set_dimension(FM::FM_3D); R_mat_grain.clear();
  if(p.isType<Teuchos::ParameterList>("Material Basis")){
    const Teuchos::ParameterList& pBasis = p.get<Teuchos::ParameterList>("Material Basis");
    FM::parseBasis(pBasis,R_mat_grain);
  } else {
    R_mat_grain(0,0) = 1.0; R_mat_grain(1,1) = 1.0; R_mat_grain(2,2) = 1.0;
  }


  // PARSE INITIAL BIN FRACTIONS
  //
  Teuchos::Array<RealType>&
    initialBinFractions = this->getInitialBinFractions();
  //Teuchos::Array<RealType> initialBinFractions;
  if(p.isType<Teuchos::Array<RealType>>("Bin Fractions") )
    initialBinFractions = p.get<Teuchos::Array<RealType>>("Bin Fractions");
  else
    initialBinFractions.resize(0);


  // PARSE PHASES
  //
  //Symmetry to phase Rotation (Rotation of inputs into correct phase symmetry, typically transverse isotropic) default [1 0 0; 0 1 0; 0 0 1]
  minitensor::Tensor<RealType, FM::FM_3D> R_phase_sym;   
  R_phase_sym.set_dimension(FM::FM_3D);
  R_phase_sym.clear();
  R_phase_sym(0,0) = 1.0; R_phase_sym(1,1) = 1.0; R_phase_sym(2,2) = 1.0;
  
     
  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >& crystalPhases = this->getCrystalPhases();
  nphases = p.get<int>("Number of Phases");
  for(int i=0; i<nphases; i++){
    const Teuchos::ParameterList& pParam = p.get<Teuchos::ParameterList>(FM::strint("Phase",i+1));
    minitensor::Tensor4<RealType, FM::FM_3D> C;   FM::parseTensor4(pParam, C  );
    minitensor::Tensor3<RealType, FM::FM_3D> ep;   FM::parseTensor3(pParam, ep  );
    minitensor::Tensor <RealType, FM::FM_3D> k; FM::parseTensor (pParam, k);
    crystalPhases.push_back(Teuchos::rcp(new FM::CrystalPhase(R_phase_sym, C, ep, k)));
  }

  // PARSE VARIANTS
  //
  Teuchos::Array<FM::CrystalVariant>& crystalVariants = this->getCrystalVariants();
  if(initialBinFractions.size() > 0){
    const Teuchos::ParameterList& vParams = p.get<Teuchos::ParameterList>("Variants");
    nvars = vParams.get<int>("Number of Variants");
    crystalVariants.resize(nvars);
    TEUCHOS_TEST_FOR_EXCEPTION(initialBinFractions.size() != nvars, std::invalid_argument,
       ">>> ERROR (FerroicModel): 'Number of Variants' must equal length of 'Bin Fractions' array");
    for(int i=0; i<nvars; i++){
      const Teuchos::ParameterList& vParam = vParams.get<Teuchos::ParameterList>(FM::strint("Variant",i+1));
      crystalVariants[i] = parseCrystalVariant(crystalPhases, vParam);
    }
  } else {
    // no variants specified.  Create single dummy variant.
    nvars = 0;
    initialBinFractions.resize(1);
    initialBinFractions[0] = 1.0;
  }
  
  
  // PARSE GRAINS
  Teuchos::Array<FM::CrystalGrain>& crystalGrains = this->getCrystalGrains();
  
  const Teuchos::ParameterList& gParams = p.get<Teuchos::ParameterList>("Grains");
  ngrains = gParams.get<int>("Number of Grains");
  
  crystalGrains.resize(ngrains);
  
  TEUCHOS_TEST_FOR_EXCEPTION(ngrains < 1, std::invalid_argument,
       ">>> ERROR (FerroicModel): 'Number of Grains' must be greater than 0");
       
  for(int i=0; i<ngrains; i++){
    crystalGrains[i] = parseCrystalGrain(crystalVariants, i, ngrains, R_mat_grain, nvars);
  }    
      
  
  Teuchos::Array<int>& nVals = this->getNVals();
  nVals.resize(3);
  nVals[0] = ngrains;
  nVals[1] = nvars;
  nVals[2] = nphases;
  

  // PARSE CRITICAL ENERGIES
  //
  
  Teuchos::Array<RealType>& tBarrier = this->getTransitionBarrier();
  tBarrier.resize(nvars*nvars);
  Teuchos::Array<RealType>& dBarrier = this->getDeltaBarrier();
  dBarrier.resize(nvars*nvars);
  Teuchos::Array<RealType>& polN = this->getPolyN();
  polN.resize(nvars*nvars);
  
  
  if(p.isType<Teuchos::ParameterList>("Critical Values")){
    const Teuchos::ParameterList& cParams = p.get<Teuchos::ParameterList>("Critical Values");
    int transitionIndex = 0;
    for(int i=0; i<nvars; i++){
      Teuchos::Array<RealType> array = cParams.get<Teuchos::Array<RealType>>(FM::strint("Variant",i+1));
      TEUCHOS_TEST_FOR_EXCEPTION(array.size()!=nvars, std::invalid_argument,
         ">>> ERROR (FerroicModel): List of critical values for variant " << i+1 << " is wrong length");
      for(int j=0; j<nvars; j++){
        tBarrier[transitionIndex] = array[j];
        transitionIndex++;
      }
    }
  }
  
  if(p.isType<Teuchos::ParameterList>("Delta Values")){
    const Teuchos::ParameterList& dB_Params = p.get<Teuchos::ParameterList>("Delta Values");
    int transitionIndex = 0;
    for(int i=0; i<nvars; i++){
      Teuchos::Array<RealType> dB_array = dB_Params.get<Teuchos::Array<RealType>>(FM::strint("Variant",i+1));
      TEUCHOS_TEST_FOR_EXCEPTION(dB_array.size()!=nvars, std::invalid_argument,
         ">>> ERROR (FerroicModel): List of delta values for variant " << i+1 << " is wrong length");
      for(int j=0; j<nvars; j++){
        dBarrier[transitionIndex] = dB_array[j];
        transitionIndex++;
      }
    }
  }
  
  if(p.isType<Teuchos::ParameterList>("PolyN Values")){
    const Teuchos::ParameterList& pN_Params = p.get<Teuchos::ParameterList>("PolyN Values");
    int transitionIndex = 0;
    for(int i=0; i<nvars; i++){
      Teuchos::Array<RealType> pN_array = pN_Params.get<Teuchos::Array<RealType>>(FM::strint("Variant",i+1));
      TEUCHOS_TEST_FOR_EXCEPTION(pN_array.size()!=nvars, std::invalid_argument,
         ">>> ERROR (FerroicModel): List of polyN values for variant " << i+1 << " is wrong length");
      for(int j=0; j<nvars; j++){
        polN[transitionIndex] = pN_array[j];
        transitionIndex++;
      }
    }
  }
  
  // PARSE INITIAL POLING FIELD
  minitensor::Vector<RealType, FM::FM_3D>& E_init = this->getEInitial();
  const Teuchos::ParameterList& eParams = p.get<Teuchos::ParameterList>("PreTreatment");
  Teuchos::Array<RealType> Earray = eParams.get<Teuchos::Array<RealType>>("Electric Field");
  E_init[0] = Earray[0];
  E_init[1] = Earray[1];
  E_init[2] = Earray[2];
  
  minitensor::Vector<RealType, FM::FM_SCAL>& RelaxFactor = this->getRelaxFactor();   //VALUE FROM 0 to 1
  if(eParams.isType<RealType>("Relax Factor") ) {
    RelaxFactor[0] = eParams.get<RealType>("Relax Factor");
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(0<1, std::invalid_argument,
         ">>> ERROR (FerroicModel): Unspecified Relax Factor");
    //RelaxFactor[0] = 0.0; //default
  }
  
  
  
  
}


/******************************************************************************/
template<typename EvalT>
void FerroicModel<EvalT>::
computeState(
      const minitensor::Tensor<ScalarT, FM::FM_3D>    & x,
      const minitensor::Vector<ScalarT, FM::FM_3D>    & E,
      const Teuchos::Array<RealType>                 & oldfractions,
            minitensor::Tensor<ScalarT, FM::FM_3D>    & X,
            minitensor::Vector<ScalarT, FM::FM_3D>    & D,
            Teuchos::Array<ScalarT>& newfractions)
/******************************************************************************/
{
  int ngrains = nVals[0];
  int nvars = nVals[1];
  int ntrans = nvars*nvars; 
  
  //Teuchos::Array<RealType> temp_oldfractions;
  
  


  // create non-linear system
  //
  using NLS = FM::DomainSwitching<EvalT>;
  NLS domainSwitching(crystalGrains,
                      tBarriers,
                      dBarriers, 
                      polyN,
                      oldfractions, 
                      x, 
                      E, /* dt= */ 1.0, nVals);

  // solution variable
  //
  
  
  
  minitensor::Vector<ScalarT,FM::MAX_GTRN> xi;   // create solution vector with initial guess
  xi.set_dimension(ngrains*ntrans);
  xi.clear();  
  

  // solve for xi
  //

  // JR todo: don't hardwire:
  m_integrationType = FM::IntegrationType::EXPLICIT;
  // m_explicitMethod = FM::ExplicitMethod::SCALED_DESCENT;
  // m_explicitMethod = FM::ExplicitMethod::DESCENT_NORM;
  m_explicitMethod = FM::ExplicitMethod::EXPLICIT_SMOOTH;
  
  
  switch (m_integrationType){

    default:
    break;

    case FM::IntegrationType::EXPLICIT:
    {

      switch (m_explicitMethod){

        default:
        break;


        case FM::ExplicitMethod::EXPLICIT_SMOOTH:
        {
          FM::Explicit_Smooth(domainSwitching, xi);
          break;
        }
      }
      break;
    }

    case FM::IntegrationType::IMPLICIT:
    {

//  

      break;
    }
  }

  // update based on new xi values
   
  Teuchos::Array<ScalarT> g_newfractions(nvars);
  Teuchos::Array<ScalarT> g_oldfractions(nvars);
  
  minitensor::Vector<ScalarT, FM::MAX_TRNS> g_xi;
  g_xi.set_dimension(ntrans); 
  
  
  ScalarT g_vf_min;
  ScalarT g_vf_sum;
  
  ScalarT zero_tol = 1e-8;
  
  for (int h = 0; h<ngrains; h++) {
    g_newfractions.clear();
    g_oldfractions.clear();
    
    g_vf_min = 0.0;
    g_vf_sum = 0.0;
    
    const FM::CrystalGrain& cg = crystalGrains[h];
    
    g_xi.clear();
    
    //PARSE xi (material) to g_xi (grain specific)    
    for (int i = 0; i<ntrans; i++) {
      g_xi[i] = xi[h*ntrans+i];
    }
    //PARSE m_binFractions (material) to g_binFractions (grain specific)    
    for (int i = 0; i<nvars; i++) {
      g_oldfractions[i] = oldfractions[h*nvars+i];
    } 
    
    //COMPUTE CHANGE TO g_fractionsNew  
    FM::computeBinFractions(g_xi, g_newfractions, g_oldfractions, cg.aMatrix, nVals);
    
    
    //Sanitizing g_newfractions:    vf(i)>0     Sum(vf) =1;
    for (int i = 0; i<nvars; i++) {
      if (g_vf_min > g_newfractions[i]){
        g_vf_min = g_newfractions[i];
      }
      g_vf_sum += g_newfractions[i];
    }
    
    g_vf_sum += nvars*(-g_vf_min);
    
    for (int i = 0; i<nvars; i++) {
      g_newfractions[i] = (g_newfractions[i] - g_vf_min)/g_vf_sum; 
    }
    
    
    
    
    for (int i = 0; i<nvars; i++) {
      TEUCHOS_TEST_FOR_EXCEPTION((g_newfractions[i] < -zero_tol), std::invalid_argument, ">>> ERROR (FerroicModel::Explicit_Smooth_Poling ) volfrac_i < 0, volfrac_i="<<g_newfractions[i]<<" in, GRAIN:"<<h<<" VAR:"<<i); 
      newfractions[h*nvars+i] = g_newfractions[i];
    }
      
  }
  
  
  
  //computeBinFractions(g_xi, g_fractionsNew, g_binFractions, cg.aMatrix);
  

  minitensor::Tensor<ScalarT, FM::FM_3D> linear_x;
  linear_x.clear();

  minitensor::Vector<ScalarT, FM::FM_3D> linear_D;
  minitensor::Vector<ScalarT, FM::FM_3D> E_relaxed;
  linear_D.clear();
  E_relaxed.clear();


  //FM::computeInitialState(newfractions, crystalVariants,
  //                        x, X, linear_x,
  //                        E, D, linear_D);

  //for debugging only
//  if (E[2]>4e-3) {
//     int breakflag = 1;    
//  }

  FM::computeFinalState(newfractions,       //const
                        crystalGrains,      //const
                        x,                  //const
                        X,                      //out
                        linear_x,               //out
                        E,                  //const
                        D,                      //out
                        linear_D,               //out
                        nVals);             //const
                          
                    

  //FM::computeRelaxedState(newfractions, crystalVariants,
  //                        x, X, linear_x,
  //                        E_relaxed, D, linear_D);




}



/******************************************************************************/
template<typename EvalT>
void FerroicModel<EvalT>::
computePolingState(
      const minitensor::Tensor<ScalarT, FM::FM_3D>& x,
      const minitensor::Vector<ScalarT, FM::FM_3D>& E,
      const Teuchos::Array<RealType>& oldfractions,
            minitensor::Tensor<ScalarT, FM::FM_3D>& X,
            minitensor::Vector<ScalarT, FM::FM_3D>& D,
            Teuchos::Array<ScalarT>& newfractions)
/******************************************************************************/
{

  // create non-linear system
  //
  using NLS = FM::DomainSwitching<EvalT>;
  NLS domainSwitching(crystalGrains,
                      tBarriers,
                      dBarriers, 
                      polyN,
                      oldfractions, 
                      x, 
                      E, /* dt= */ 1.0, nVals);

  // solution variable
  //
  
  int ngrains = nVals[0];
  int nvars = nVals[1];
  int ntrans = nvars*nvars; //domainSwitching.getNumStates();
  
  minitensor::Vector<ScalarT,FM::MAX_GTRN> xi;   // create solution vector with initial guess
  xi.set_dimension(ngrains*ntrans);
  xi.clear();  
  

  // solve for xi
  
  FM::Explicit_Smooth_Poling(domainSwitching, xi);
  

  Teuchos::Array<ScalarT> g_newfractions(nvars);
  Teuchos::Array<ScalarT> g_oldfractions(nvars);
  
  minitensor::Vector<ScalarT, FM::MAX_TRNS> g_xi;
  g_xi.set_dimension(ntrans); 
  
  
  ScalarT zero_tol = 1e-8;
  
  for (int h = 0; h<ngrains; h++) {
    g_newfractions.clear();
    g_oldfractions.clear();
    
    const FM::CrystalGrain& cg = crystalGrains[h];
    
    g_xi.clear();
    
    //PARSE xi (material) to g_xi (grain specific)    
    for (int i = 0; i<ntrans; i++) {
      g_xi[i] = xi[h*ntrans+i];
    }
    //PARSE m_binFractions (material) to g_binFractions (grain specific)    
    for (int i = 0; i<nvars; i++) {
      g_oldfractions[i] = oldfractions[h*nvars+i];
    } 
    
    //COMPUTE CHANGE TO g_fractionsNew  
    FM::computeBinFractions(g_xi, g_newfractions, g_oldfractions, cg.aMatrix, nVals);
    
    
    
    
    
    
    for (int i = 0; i<nvars; i++) {
      TEUCHOS_TEST_FOR_EXCEPTION((g_newfractions[i] < -zero_tol), std::invalid_argument, ">>> ERROR (FerroicModel::Explicit_Smooth_Poling ) volfrac_i < 0, volfrac_i="<<g_newfractions[i]<<" in, GRAIN:"<<h<<" VAR:"<<i); 
      newfractions[h*nvars+i] = g_newfractions[i];
    }
      
  }
  
  
  minitensor::Tensor<ScalarT, FM::FM_3D> linear_x;
  linear_x.clear();

  minitensor::Vector<ScalarT, FM::FM_3D> linear_D;
  minitensor::Vector<ScalarT, FM::FM_3D> E_relaxed;
  linear_D.clear();
  E_relaxed.clear();




  FM::computeFinalState(newfractions,       //const
                        crystalGrains,      //const
                        x,                  //const
                        X,                      //out
                        linear_x,               //out
                        E,                  //const
                        D,                      //out
                        linear_D,               //out
                        nVals);             //const
 

}






/******************************************************************************/
template<typename EvalT>
void FerroicModel<EvalT>::
PrePolingRoutine(
                Teuchos::Array<RealType>        & newfractions)   //return
/******************************************************************************/
{
  
  const Teuchos::Array<int>& nVals = this->getNVals();
  int nGrains = nVals[0];  
  int nVariants = nVals[1];
  
  //int nPhases = nVals[2];         //unused in this function
  
  
  minitensor::Tensor<ScalarT,FM::FM_3D> X, x;
  minitensor::Vector<ScalarT,FM::FM_3D> E, D;
  X.clear();
  x.clear();
  
  E.clear();
  D.clear();
  
  
  
  //START: pre-poling section --------------------------------------------------
  Teuchos::Array<RealType> oldfractions(nGrains*nVariants);
  const Teuchos::Array<ScalarT>& binfractions = this->getInitialBinFractions();
    
  const minitensor::Vector<RealType, FM::FM_3D>& E_init = this->getEInitial();       //Write INITIALBINFRACTIONS (PREPOLE) to OLDFRACTIONS
  E = E_init;
  //E[0] = E_init[0];
  //E[1] = E_init[1];
  //E[2] = E_init[2];
  
  for(int h=0; h<nGrains; h++){
    for(int i=0; i<nVariants; i++){
        oldfractions[h*nVariants+i] = binfractions[i];
        
    }
  }
  
  int poling_ctr_limit = 100;
  
  
  
  //POLE TO E_POLING
  for (int ip = 0; ip < poling_ctr_limit; ip++) {
    
    this->computePolingState( 
                                x,                //const
                                E,                //const
                                oldfractions,     //const
                                X, 
                                D, 
                                newfractions); 
                                
    for(int h=0; h<nGrains; h++){
      for(int i=0; i<nVariants; i++){
        oldfractions[h*nVariants+i] = newfractions[h*nVariants+i];
      }
    }
    
  }// end FOR ip
  
    
  
  
  E.clear();   //zero electric field
  
  //POLE BACK TO 0 E.Field.
  for (int ip = 0; ip < poling_ctr_limit; ip++) {
  
    this->computePolingState( 
                                x,                //const
                                E,                //const
                                oldfractions,     //const
                                X, 
                                D, 
                                newfractions);
  
    for(int h=0; h<nGrains; h++){
      for(int i=0; i<nVariants; i++){
        oldfractions[h*nVariants+i] = newfractions[h*nVariants+i];
      }
    }
    
  }//end FOR ip
  
  //END:   pre-poling section --------------------------------------------------
    


} //END PREPOLINGROUTINE









}
