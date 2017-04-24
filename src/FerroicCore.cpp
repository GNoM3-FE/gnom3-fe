#include "FerroicCore.hpp"

/******************************************************************************/
FM::CrystalPhase::
CrystalPhase(Intrepid2::Tensor <RealType, FM::FM_3D>& matBasis,
             Intrepid2::Tensor4<RealType, FM::FM_3D>& C_matBasis,
             Intrepid2::Tensor3<RealType, FM::FM_3D>& ep_matBasis,
             Intrepid2::Tensor <RealType, FM::FM_3D>& k_matBasis)
/******************************************************************************/
{
  FM::changeBasis(C, C_matBasis, matBasis);
  FM::changeBasis(ep, ep_matBasis, matBasis);
  FM::changeBasis(k, k_matBasis, matBasis);


  basis = matBasis;
}

namespace FM {

std::string strint(std::string s, int i, char delim) {
    std::ostringstream ss;
    ss << s << delim << i;
    return ss.str();
}

}
