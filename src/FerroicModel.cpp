#include "FerroicModel.hpp"

namespace FM {

/******************************************************************************/
void parseBasis(const Teuchos::ParameterList& pBasis,
                      Intrepid2::Tensor<RealType, FM::FM_3D>& basis)
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
                   Intrepid2::Tensor4<RealType, FM::FM_3D>& C)
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
                   Intrepid2::Tensor3<RealType, FM::FM_3D>& ep)
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
                   Intrepid2::Tensor<RealType, FM::FM_3D>& k)
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

  TEUCHOS_TEST_FOR_EXCEPTION(phaseIndex < 0 || phaseIndex >= phases.size(),
    std::invalid_argument,
    ">>> ERROR (FerroicModel): Requested phase has not been defined.");


  if(vParam.isType<Teuchos::ParameterList>("Crystallographic Basis")){
    cv.basis.set_dimension(phases[phaseIndex]->C.get_dimension());
    const Teuchos::ParameterList&
    pBasis = vParam.get<Teuchos::ParameterList>("Crystallographic Basis");
    FM::parseBasis(pBasis,cv.basis);
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
      cv.spontStrain(1,2) = inVals[3];
      cv.spontStrain(0,2) = inVals[4];
      cv.spontStrain(0,1) = inVals[5];
      cv.spontStrain(2,1) = inVals[3];
      cv.spontStrain(2,0) = inVals[4];
      cv.spontStrain(1,0) = inVals[5];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous Strain'.");


  cv.C.set_dimension(phases[phaseIndex]->C.get_dimension()); cv.C.clear();
  FM::changeBasis(cv.C, phases[phaseIndex]->C, cv.basis);

  cv.ep.set_dimension(phases[phaseIndex]->ep.get_dimension()); cv.ep.clear();
  FM::changeBasis(cv.ep, phases[phaseIndex]->ep, cv.basis);

  cv.k.set_dimension(phases[phaseIndex]->k.get_dimension()); cv.k.clear();
  FM::changeBasis(cv.k, phases[phaseIndex]->k, cv.basis);

  return cv;
}

}
