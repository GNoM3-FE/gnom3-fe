//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//


#include "FerroicModel.hpp"

namespace FM {

/******************************************************************************/
void parseBasis(const Teuchos::ParameterList& pBasis,
                      minitensor::Tensor<RealType, FM::FM_3D>& basis)
/******************************************************************************/
{
  if(pBasis.isType<Teuchos::Array<RealType>>("X axis")){
    Teuchos::Array<RealType> Xhat = pBasis.get<Teuchos::Array<RealType>>("X axis");
    basis(0,0) = Xhat[0]; basis(1,0) = Xhat[1]; basis(2,0) = Xhat[2];
  }
  if(pBasis.isType<Teuchos::Array<RealType>>("Y axis")){
    Teuchos::Array<RealType> Yhat = pBasis.get<Teuchos::Array<RealType>>("Y axis");
    basis(0,1) = Yhat[0]; basis(1,1) = Yhat[1]; basis(2,1) = Yhat[2];
  }
  if(pBasis.isType<Teuchos::Array<RealType>>("Z axis")){
    Teuchos::Array<RealType> Zhat = pBasis.get<Teuchos::Array<RealType>>("Z axis");
    basis(0,2) = Zhat[0]; basis(1,2) = Zhat[1]; basis(2,2) = Zhat[2];
  }
}

/******************************************************************************/
void
parseTensor4(const Teuchos::ParameterList& cParam,
                   minitensor::Tensor4<RealType, FM::FM_3D>& C)
/******************************************************************************/
{

  // JR:  This should be generalized to read stiffness tensors of various
  // symmetries.

  // parse
  //
  RealType C11 = cParam.get<RealType>("C11");
  RealType C33 = cParam.get<RealType>("C33");
  RealType C12 = cParam.get<RealType>("C12");
  RealType C23 = cParam.get<RealType>("C23");
  RealType C44 = cParam.get<RealType>("C44");
  RealType C66 = cParam.get<RealType>("C66");

  C.clear();

  C(0,0,0,0) = C11; C(0,0,1,1) = C12; C(0,0,2,2) = C23;
  C(1,1,0,0) = C12; C(1,1,1,1) = C11; C(1,1,2,2) = C23;
  C(2,2,0,0) = C23; C(2,2,1,1) = C23; C(2,2,2,2) = C33;
  C(0,1,0,1) = C66/2.0; C(1,0,1,0) = C66/2.0;
  C(0,2,0,2) = C44/2.0; C(2,0,2,0) = C44/2.0;
  C(1,2,1,2) = C44/2.0; C(2,1,2,1) = C44/2.0;
}
/******************************************************************************/
void
parseTensor3(const Teuchos::ParameterList& cParam,
                   minitensor::Tensor3<RealType, FM::FM_3D>& ep)
/******************************************************************************/
{
  // JR:  This should be generalized to read piezoelectric tensors of various
  // symmetries.

  // parse
  //
  RealType ep31 = cParam.get<RealType>("ep31");
  RealType ep33 = cParam.get<RealType>("ep33");
  RealType ep15 = cParam.get<RealType>("ep15");

  ep.clear();
  ep(0,0,2) = ep15/2.0; ep(0,2,0) = ep15/2.0;
  ep(1,1,2) = ep15/2.0; ep(1,2,1) = ep15/2.0;
  ep(2,0,0) = ep31; ep(2,1,1) = ep31; ep(2,2,2) = ep33;
}
/******************************************************************************/
void
parseTensor(const Teuchos::ParameterList& cParam,
                   minitensor::Tensor<RealType, FM::FM_3D>& k)
/******************************************************************************/
{
  // JR:  This should be generalized to read permittivity tensors of various
  // symmetries.

  // parse
  //
  RealType k11 = cParam.get<RealType>("Eps11");
  RealType k33 = cParam.get<RealType>("Eps33");

  k.clear();
  k(0,0) = k11;
  k(1,1) = k11;
  k(2,2) = k33;
}


/******************************************************************************/
FM::CrystalVariant
parseCrystalVariant(const Teuchos::Array<Teuchos::RCP<FM::CrystalPhase>>& phases,
                    const Teuchos::ParameterList& vParam)
/******************************************************************************/
{

  TEUCHOS_TEST_FOR_EXCEPTION(phases.size()==0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalVariant constructor passed empty list of Phases.");

  FM::CrystalVariant cv;

  int phaseIndex;
  if(vParam.isType<int>("Phase")){
    phaseIndex = vParam.get<int>("Phase") ;
    phaseIndex--; // Ids are one-based.  Indices are zero-based.
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a phase.");
    
  cv.phaseIndex = phaseIndex;

  TEUCHOS_TEST_FOR_EXCEPTION(phaseIndex < 0 || phaseIndex >= phases.size(),
    std::invalid_argument,
    ">>> ERROR (FerroicModel): Requested phase has not been defined.");


  if(vParam.isType<Teuchos::ParameterList>("Crystallographic Basis")){
    cv.R_variant_phase.set_dimension(phases[phaseIndex]->C.get_dimension());
    const Teuchos::ParameterList&
    pBasis = vParam.get<Teuchos::ParameterList>("Crystallographic Basis");
    FM::parseBasis(pBasis,cv.R_variant_phase);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a crystallograph basis.");

  if(vParam.isType<Teuchos::Array<RealType>>("Spontaneous Polarization")){
    Teuchos::Array<RealType>
      inVals = vParam.get<Teuchos::Array<RealType>>("Spontaneous Polarization");
      TEUCHOS_TEST_FOR_EXCEPTION(inVals.size() != FM::FM_3D, std::invalid_argument,
      ">>> ERROR (FerroicModel): Expected 3 terms 'Spontaneous Polarization' vector.");
      cv.spontEDisp.set_dimension(FM::FM_3D);
      for(int i=0; i<FM::FM_3D; i++) cv.spontEDisp(i) = inVals[i];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous Polarization'.");

  if(vParam.isType<Teuchos::Array<RealType>>("Spontaneous Strain")){
    Teuchos::Array<RealType>
      inVals = vParam.get<Teuchos::Array<RealType>>("Spontaneous Strain");
      TEUCHOS_TEST_FOR_EXCEPTION(inVals.size() != 6, std::invalid_argument,
      ">>> ERROR (FerroicModel): Expected 6 voigt terms 'Spontaneous Strain' tensor.");
      cv.spontStrain.set_dimension(FM::FM_3D);
      cv.spontStrain(0,0) = inVals[0];
      cv.spontStrain(1,1) = inVals[1];
      cv.spontStrain(2,2) = inVals[2];
      cv.spontStrain(1,2) = inVals[3]/2.0;
      cv.spontStrain(0,2) = inVals[4]/2.0;
      cv.spontStrain(0,1) = inVals[5]/2.0;
      cv.spontStrain(2,1) = inVals[3]/2.0;
      cv.spontStrain(2,0) = inVals[4]/2.0;
      cv.spontStrain(1,0) = inVals[5]/2.0;
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous Strain'.");


  cv.C.set_dimension(phases[phaseIndex]->C.get_dimension()); cv.C.clear();
  FM::changeBasis(cv.C, phases[phaseIndex]->C, cv.R_variant_phase);

  cv.ep.set_dimension(phases[phaseIndex]->ep.get_dimension()); cv.ep.clear();
  FM::changeBasis(cv.ep, phases[phaseIndex]->ep, cv.R_variant_phase);

  cv.k.set_dimension(phases[phaseIndex]->k.get_dimension()); cv.k.clear();
  FM::changeBasis(cv.k, phases[phaseIndex]->k, cv.R_variant_phase);

  return cv;
}



/******************************************************************************/
FM::CrystalGrain
parseCrystalGrain(const Teuchos::Array<FM::CrystalVariant>& variants,
                  const int igrains,
                  const int ngrains,
                  const minitensor::Tensor<RealType, FM::FM_3D>& R_mat_grain,
                  const int nvars)

/******************************************************************************/
{
  //INPUT ERROR CHECKS  
  TEUCHOS_TEST_FOR_EXCEPTION(variants.size()==0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalGrain constructor passed empty list of Variants.");
  TEUCHOS_TEST_FOR_EXCEPTION(nvars==0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalGrain constructor passed 0 value nvars.");
  TEUCHOS_TEST_FOR_EXCEPTION(igrains<0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalGrain constructor passed invalid grain index.");
  TEUCHOS_TEST_FOR_EXCEPTION(ngrains<=0, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalGrain constructor passed 0 value ngrain.");
  TEUCHOS_TEST_FOR_EXCEPTION(igrains>=ngrains, std::invalid_argument,
  ">>> ERROR (FerroicModel): CrystalGrain constructor passed grain index greater than ngrains.");

  FM::CrystalGrain cg;
  
    
  // CREATE R_GRAIN_VARIANT (rotation to grain from variant)
    

  cg.R_grain_variant.set_dimension(FM::FM_3D);
  cg.R_grain_variant.clear();
  
  RealType grain_theta = igrains*2.39996322972865332;
  RealType grain_phi = acos(1.0 - 2.0*igrains/ngrains);
  
  cg.R_grain_variant(0,0)=cos(grain_theta)*cos(grain_phi);
  cg.R_grain_variant(0,1)=-sin(grain_theta);
  cg.R_grain_variant(0,2)=cos(grain_theta)*sin(grain_phi);
  
  cg.R_grain_variant(1,0)=sin(grain_theta)*cos(grain_phi);
  cg.R_grain_variant(1,1)=cos(grain_theta);
  cg.R_grain_variant(1,2)=sin(grain_theta)*sin(grain_phi);
  
  cg.R_grain_variant(2,0)=-sin(grain_phi);
  cg.R_grain_variant(2,1)=0.0;
  cg.R_grain_variant(2,2)=cos(grain_phi);
  
  
  


  // PARSE CRYSTALVARIANT INTO GRAIN_VARIANT
  
  cg.crystalVariants.resize(nvars);
  
  
  minitensor::Tensor4<RealType, FM_3D> dum_C;
  dum_C.set_dimension(FM::FM_3D);
  dum_C.clear();
  
  minitensor::Tensor3<RealType, FM_3D> dum_ep;
  dum_ep.set_dimension(FM::FM_3D);
  dum_ep.clear();
  
  minitensor::Tensor<RealType, FM_3D> dum_k;
  dum_k.set_dimension(FM::FM_3D);
  dum_k.clear();
  
  minitensor::Tensor<RealType, FM_3D> dum_R; //change to R_phase_variant 
  dum_R.set_dimension(FM::FM_3D);
  dum_R.clear();
  
  minitensor::Tensor<RealType, FM_3D> dum_spontS;
  dum_spontS.set_dimension(FM::FM_3D);
  dum_spontS.clear();
  
  minitensor::Vector<RealType, FM_3D> dum_spontD;
  dum_spontD.set_dimension(FM::FM_3D);
  dum_spontD.clear();
  
  
        
  for (int i = 0; i < nvars; i++){
      
    //check later to see if this is . or -> (pointer function call)
    
    const FM::CrystalVariant& cv = variants[i];
    FM::CrystalVariant& cg_cv = cg.crystalVariants[i];
      
    cg_cv.R_variant_phase.set_dimension(FM::FM_3D);
    cg_cv.R_variant_phase.clear();
    cg_cv.R_variant_phase = cv.R_variant_phase;
    
    
    cg_cv.C.set_dimension(FM::FM_3D);
    dum_C.clear();
    cg_cv.C.clear();
    FM::changeBasis(dum_C, cv.C, cg.R_grain_variant);
    FM::changeBasis(cg_cv.C, dum_C, R_mat_grain);
    
    
    cg_cv.ep.set_dimension(FM::FM_3D);
    dum_ep.clear();
    cg_cv.ep.clear();
    FM::changeBasis(dum_ep, cv.ep, cg.R_grain_variant);
    FM::changeBasis(cg_cv.ep, dum_ep, R_mat_grain);
    
    cg_cv.k.set_dimension(FM::FM_3D);
    dum_k.clear();
    cg_cv.k.clear();
    FM::changeBasis(dum_k, cv.k, cg.R_grain_variant);
    FM::changeBasis(cg_cv.k, dum_k, R_mat_grain);
    
    
    cg_cv.spontStrain.set_dimension(FM::FM_3D);
    dum_spontS.clear();
    cg_cv.spontStrain.clear();
    FM::changeBasis(dum_spontS, cv.spontStrain, cg.R_grain_variant);
    FM::changeBasis(cg_cv.spontStrain, dum_spontS, R_mat_grain);
    
    cg_cv.spontEDisp.set_dimension(FM::FM_3D);
    dum_spontD.clear();
    cg_cv.spontEDisp.clear();
    FM::changeBasis(dum_spontD, cv.spontEDisp, cg.R_grain_variant);
    FM::changeBasis(cg_cv.spontEDisp, dum_spontD, R_mat_grain);
    
    cg_cv.phaseIndex = cv.phaseIndex;
              
  }    
          
      
    
  // PARSE CRYSTALVARIANT INTO TRANSITION
    

  int ntrans = nvars*nvars;
  
  cg.transitions.resize(ntrans);
  
  int transIndex = 0;
  for(int I=0; I<nvars; I++){
    for(int J=0; J<nvars; J++){
        
      FM::Transition& t = cg.transitions[transIndex];
      
      t.transStrain.clear();
      t.transEDisp.clear();
      
      
      CrystalVariant& fromVariant = cg.crystalVariants[I];
      CrystalVariant& toVariant = cg.crystalVariants[J];
      t.transStrain = toVariant.spontStrain - fromVariant.spontStrain;
      t.transEDisp  = toVariant.spontEDisp  - fromVariant.spontEDisp;
      transIndex++;
    }
  }
    
    
  // PARSE aMatrix
  
  
  cg.aMatrix.set_dimension(nvars*ntrans);   // nvars x ntrans, i = ivars*nvars + itrans
  cg.aMatrix.clear();
  
  int ivars=0;
  int itrans=0;
  
  for(int I=0; I<nvars; I++){
    for(int J=0; J<nvars; J++){
      
      ivars = I;
      itrans = nvars*I+J;  
      cg.aMatrix(ivars*ntrans+itrans) = -1.0;
      
      ivars = J;
      itrans = nvars*I+J;
      cg.aMatrix(ivars*ntrans+itrans) = 1.0;
    }
    
    ivars = I;
    itrans = nvars*I+I;
    cg.aMatrix(ivars*ntrans+itrans) = 0.0;
  }

  return cg;
}



}
