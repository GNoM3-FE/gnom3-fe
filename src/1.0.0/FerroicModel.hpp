//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//




#if !defined(FerroicModel_hpp)
#define FerroicModel_hpp


#include "Sacado.hpp"
#include "MiniTensor.h"

#include "FerroicCore.hpp"

namespace FM
{

template<typename EvalT>
class FerroicModel
{
public:

  using ScalarT = typename EvalT::ScalarT;

  // Constructor
  //
  FerroicModel();

  // Virtual Denstructor
  //
  ~FerroicModel()
  {
  }

  // Method to compute the state
  //
  void
  computeState(
      const minitensor::Tensor<ScalarT, FM::FM_3D>   & x,
      const minitensor::Vector<ScalarT, FM::FM_3D>   & E,
      const Teuchos::Array<RealType>                & oldfractions,
            minitensor::Tensor<ScalarT, FM::FM_3D>   & X,
            minitensor::Vector<ScalarT, FM::FM_3D>   & D,
            Teuchos::Array<ScalarT>                 & newfractions);
            
            
  void
  computePolingState(
      const minitensor::Tensor<ScalarT, FM::FM_3D>& x,
      const minitensor::Vector<ScalarT, FM::FM_3D>& E,
      const Teuchos::Array<RealType>& oldfractions,
            minitensor::Tensor<ScalarT, FM::FM_3D>& X,
            minitensor::Vector<ScalarT, FM::FM_3D>& D,
            Teuchos::Array<ScalarT>& newfractions);

  void
  Parse(const Teuchos::ParameterList& p);
  
  

  void
  PrePolingRoutine(
            Teuchos::Array<RealType>        & newfractions);


  
  // Accessors
  //
  minitensor::Tensor<RealType, FM::FM_3D>&
  getBasis() { return R_mat_grain; }

  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >&
  getCrystalPhases() { return crystalPhases; }

  Teuchos::Array<FM::CrystalVariant>&
  getCrystalVariants() { return crystalVariants; }
  
  Teuchos::Array<FM::CrystalGrain>&
  getCrystalGrains() { return crystalGrains; }
  
  Teuchos::Array<RealType>&
  getInitialBinFractions() { return initialBinFractions; }

  //std::vector< FM::Transition >&
  //getTransitions() { return transitions; }

  Teuchos::Array<RealType>&
  getTransitionBarrier() { return tBarriers; }
  
  Teuchos::Array<RealType>&
  getDeltaBarrier() { return dBarriers; }
  
  Teuchos::Array<RealType>&
  getPolyN() { return polyN; }
  
  
  Teuchos::Array<int>&
  getNVals() {return nVals; }
  
  
  minitensor::Vector<RealType, FM::FM_3D>&
  getEInitial() { return E_initial; }
  
  
  minitensor::Vector<RealType, FM::FM_SCAL>&
  getRelaxFactor() {return RelaxFactor; };

  //void PostParseInitialize();

private:

  ///
  /// Private to prohibit copying
  ///
  FerroicModel(const FerroicModel&);

  ///
  /// Private to prohibit copying
  ///
  FerroicModel& operator=(const FerroicModel&);

  // parameters
  //
  minitensor::Tensor<RealType, FM::FM_3D>          R_mat_grain;  //material to global  //basis; 
  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> > crystalPhases;        //Reference phases for creating crystalVariant
  Teuchos::Array<FM::CrystalVariant>              crystalVariants;      //Reference variants for creating crystalGrains
  Teuchos::Array<FM::CrystalGrain>                crystalGrains;
  Teuchos::Array<RealType>                        tBarriers;            //transformation energy barriers
  Teuchos::Array<RealType>                        dBarriers;            //width of transformation energy barriers
  Teuchos::Array<RealType>                        polyN;                //polynomial order of transformation barrier sharpness
  Teuchos::Array<RealType>                        initialBinFractions;  //initial Bin Fractions (unpoled usually)
  minitensor::Vector<RealType, FM::FM_3D>          E_initial;            //Poling Field
  Teuchos::Array<int>                             nVals;  //ngrains, nvars, nphases
  minitensor::Vector<RealType, FM::FM_SCAL>        RelaxFactor;
  

  // Solution options
  //
  IntegrationType       m_integrationType;
  ExplicitMethod        m_explicitMethod;
  minitensor::StepType   m_step_type;

  RealType              m_implicit_nonlinear_solver_relative_tolerance_;
  RealType              m_implicit_nonlinear_solver_absolute_tolerance_;
  int                   m_implicit_nonlinear_solver_max_iterations_;
  int                   m_implicit_nonlinear_solver_min_iterations_;

};











/* !!!! */
void parseBasis  (const Teuchos::ParameterList&   pBasis,
                        minitensor::Tensor <RealType, FM::FM_3D>& basis);
void parseTensor4(const Teuchos::ParameterList&   pConsts,
                        minitensor::Tensor4<RealType, FM::FM_3D>& tensor);
void parseTensor3(const Teuchos::ParameterList&   pConsts,
                        minitensor::Tensor3<RealType, FM::FM_3D>& tensor);
void parseTensor (const Teuchos::ParameterList&   pConsts,
                        minitensor::Tensor <RealType, FM::FM_3D>& tensor);

FM::CrystalVariant
parseCrystalVariant(const Teuchos::Array<Teuchos::RCP<FM::CrystalPhase>>& phases,
                    const Teuchos::ParameterList& vParam);
                    
                    
FM::CrystalGrain
parseCrystalGrain(const Teuchos::Array<FM::CrystalVariant>& variants,
                  const int i,
                  const int ngrains,
                  const minitensor::Tensor<RealType, FM::FM_3D>& R_mat_grain,
                  const int nvars);
                  

}

#include "FerroicModel_Def.hpp"

#endif

