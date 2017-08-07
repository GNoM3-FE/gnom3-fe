//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//
//
//



#include "FerroicCore.hpp"

/******************************************************************************/
FM::CrystalPhase::
CrystalPhase(minitensor::Tensor <RealType, FM::FM_3D>& R_sym,
             minitensor::Tensor4<RealType, FM::FM_3D>& C_sym,
             minitensor::Tensor3<RealType, FM::FM_3D>& ep_sym,
             minitensor::Tensor <RealType, FM::FM_3D>& k_sym)
/******************************************************************************/
{
  FM::changeBasis(C, C_sym, R_sym);
  FM::changeBasis(ep, ep_sym, R_sym);
  FM::changeBasis(k, k_sym, R_sym);


  R_phase_sym = R_sym;
}

namespace FM {

std::string strint(std::string s, int i, char delim) {
    std::ostringstream ss;
    ss << s << delim << i;
    return ss.str();
}

}
