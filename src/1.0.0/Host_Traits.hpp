//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//


#include "code_types_HOST.h"

typedef Real RealType;

struct Host_Traits {
  struct Residual {
    typedef Real ScalarT;
  };
};

