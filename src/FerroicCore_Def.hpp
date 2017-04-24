//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


/******************************************************************************/
template<typename EvalT>
FM::DomainSwitching<EvalT>::DomainSwitching(
      Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
      Teuchos::Array<FM::Transition>      const & transitions,
      Teuchos::Array<RealType>            const & transBarriers,
      Teuchos::Array<RealType>            const & binFractions,
      Intrepid::FieldContainer<RealType>  const & aMatrix,
      Intrepid2::Tensor<ArgT,FM_3D>     const & x,
      Intrepid2::Vector<ArgT,FM_3D>     const & E,
      RealType dt)
  :
      m_crystalVariants(crystalVariants),
      m_transitions(transitions),
      m_transBarriers(transBarriers),
      m_binFractions(binFractions),
      m_aMatrix(aMatrix),
      m_x(x), m_dt(dt)
/******************************************************************************/
{

  // compute trial state
  //
  Intrepid2::Tensor<ArgT, FM::FM_3D> X, linear_x;
  Intrepid2::Vector<ArgT, FM::FM_3D> linear_D;
  FM::computeInitialState(m_binFractions, m_crystalVariants,
                          m_x,X,linear_x, E,m_D,linear_D);


  // set all transitions active for first residual eval
  //
  int nTransitions = m_transitions.size();
  m_transitionMap.resize(nTransitions);
  for(int J=0; J<nTransitions; J++){
    m_transitionMap[J] = J;
  }

  // evaluate residual at current bin fractions
  //
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> zero;
  zero.set_dimension(nTransitions);
  zero.clear();
  Intrepid2::Vector<ArgT, FM::MAX_TRNS>
  residual = this->gradient(zero);

  // find active transitions
  //
  int nVariants = m_binFractions.size();
  int transition=0, nActive=0;
  for(int J=0; J<nTransitions; J++)
    m_transitionMap[J] = -1;
  for(int I=0;I<nVariants;I++){
    if(m_binFractions[I] <= 1.0e-10) continue;
    for(int J=0;J<nVariants;J++){
      transition = I*nVariants+J;
      if(residual[transition] < -0.01){
        m_transitionMap[transition] = nActive;
        nActive++;
      }
    }
  }
  m_numActiveTransitions = nActive;
}

/******************************************************************************/
template<typename EvalT>
template<typename T, Intrepid2::Index N>
T
FM::DomainSwitching<EvalT>::value(Intrepid2::Vector<T, N> const & x)
/******************************************************************************/
{
  return Intrepid2::Function_Base<
    DomainSwitching<EvalT>, typename EvalT::ScalarT>::value(*this, x);
}

/******************************************************************************/
template<typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
FM::DomainSwitching<EvalT>::gradient(
  Intrepid2::Vector<T, N> const & xi) const
/******************************************************************************/
{

  Intrepid2::Tensor<T, FM::FM_3D> X;         X.clear();
  Intrepid2::Tensor<T, FM::FM_3D> linear_x;  linear_x.clear();

  Intrepid2::Vector<T, FM::FM_3D> E;         E.clear();
  Intrepid2::Vector<T, FM::FM_3D> linear_D;  linear_D.clear();

  // apply transition increment
  //
  Teuchos::Array<T> fractionsNew(m_binFractions.size());
  computeBinFractions(xi, fractionsNew, m_binFractions, m_transitionMap, m_aMatrix);

  Intrepid2::Tensor<T, FM::FM_3D> const
  x_peeled = LCM::peel_tensor<EvalT, T, N, FM::FM_3D>()(m_x);

  Intrepid2::Vector<T, FM::FM_3D> const
  D_peeled = LCM::peel_vector<EvalT, T, N, FM::FM_3D>()(m_D);

  // compute new state
  //
  computeRelaxedState(fractionsNew, m_crystalVariants, x_peeled,X,linear_x, E,D_peeled,linear_D);

  // compute new residual
  //
  auto const num_unknowns = xi.get_dimension();
  Intrepid2::Vector<T, N> residual(num_unknowns);
  computeResidual(residual, fractionsNew,
                  m_transitionMap, m_transitions, m_crystalVariants,
                  m_transBarriers, m_aMatrix,
                  X,linear_x, E,linear_D);

  return residual;
}


/******************************************************************************/
template<typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
FM::DomainSwitching<EvalT>::updateXi(
    Intrepid2::Vector<T, N> const & xi)
/******************************************************************************/
{
  // THIS FUNCTION CONDITIONS THE Gradient SO THAT BINS REMAIN NEGATIVE (will
  // transform), IS NORMALIZED, AND SCALED BY binFractions

  auto const num_unknowns = xi.get_dimension();
  Intrepid2::Vector<T, N> new_xi(num_unknowns);
  Intrepid2::Vector<T, N> grad(num_unknowns);
  grad = FM::DomainSwitching<EvalT>::gradient(xi);

  int nVariants = m_binFractions.size();

  Teuchos::Array<T> fractionsNew(m_binFractions.size());
  computeBinFractions(xi, fractionsNew, m_binFractions, m_transitionMap, m_aMatrix);

  for (int I=0; I<nVariants; I++){
    T fracI = fractionsNew[I];

    //Sum of the Gradient along J for each I
    T SumG = 0;
    for (int J=0; J<nVariants; J++){

      //i: the transition index of transformation I->J
      int i = I*nVariants+J;
      int tm_index = m_transitionMap[i];

      if (m_transitionMap[i]>=0){
        //the transitionMap index for this transformation
        if (grad[tm_index]>0){  //Positive grad means no energy for transformation
          grad[tm_index] = 0;
        }
        SumG += grad[tm_index];
      }
    }//END FOR J

    //computes Xi from conditioned Resid only if sumG >0 (We know this is true if transitionMap[i]>0)

    for (int J=0; J<nVariants; J++){
      //i: the transition index of transformation I->J
      int i = I*nVariants+J;
      int tm_index = m_transitionMap[i];

      if (m_transitionMap[i]>=0){

        if (abs(SumG) > 0){
          new_xi[tm_index] = grad[tm_index]/SumG*fracI;
        }
        else{
          new_xi[tm_index] = 0;
        }

      }
    }// END FOR J

  }//END FOR I

  return new_xi;

}



/******************************************************************************/
template<typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Vector<T, N>
FM::DomainSwitching<EvalT>::conditionedGrad(
    Intrepid2::Vector<T, N> const & xi_new,
    Intrepid2::Vector<T, N> const & xi_old)
/******************************************************************************/
{

  auto const num_unknowns = xi_new.get_dimension();

  Intrepid2::Vector<T, N> c_grad(num_unknowns);  //Return Value

  Intrepid2::Vector<T, N> grad(num_unknowns);
  grad = FM::DomainSwitching<EvalT>::gradient(xi_new);

  int nVariants = m_binFractions.size();

  Teuchos::Array<T> fractionsOld(m_binFractions.size());
  computeBinFractions(xi_old, fractionsOld, m_binFractions, m_transitionMap, m_aMatrix);

  //computes Xi from conditioned Resid only if sumG >0 (We know this is true if transitionMap[i]>0)


  for (int I=0; I<nVariants; I++){
    T fracI = fractionsOld[I];

    for (int J=0; J<nVariants; J++){
      //i: the transition index of transformation I->J
      int i = I*nVariants+J;
      int tm_index = m_transitionMap[i];

      if (m_transitionMap[i]>=0){

          c_grad[tm_index] = grad[tm_index]*fracI;

      }
    }// END FOR J

  }//END FOR I

  return c_grad;

}



/******************************************************************************/
template<typename EvalT>
template<typename T, Intrepid2::Index N>
Intrepid2::Tensor<T, N>
FM::DomainSwitching<EvalT>::hessian(
    Intrepid2::Vector<T, N> const & xi)
/******************************************************************************/
{
//  return Intrepid2::Function_Base<DomainSwitching<EvalT>,ArgT>::hessian(*this,xi);

  using AD = Sacado::Fad::SLFad<T, N>;

  Intrepid2::Index const
  dimension = xi.get_dimension();

  Intrepid2::Vector<AD, N>
  x_ad(dimension);

  for (Intrepid2::Index i{0}; i < dimension; ++i) {
    x_ad(i) = AD(dimension, i, xi(i));
  }

  Intrepid2::Vector<AD, N> const
  r_ad = this->gradient(x_ad);

  Intrepid2::Tensor<T, N>
  Hessian(dimension);

  for (Intrepid2::Index i{0}; i < dimension; ++i) {
    for (Intrepid2::Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = r_ad(i).dx(j);
    }
  }

  return Hessian;
}




/******************************************************************************/
template<typename DataT>
void
FM::changeBasis(       Intrepid2::Tensor4<DataT, FM::FM_3D>& inGlobalBasis,
                 const Intrepid2::Tensor4<DataT, FM::FM_3D>& inMatlBasis,
                 const Intrepid2::Tensor <DataT, FM::FM_3D>& matlBasis)
/******************************************************************************/
{
    int num_dims = matlBasis.get_dimension();
    inGlobalBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int k=0; k<num_dims; k++)
       for(int l=0; l<num_dims; l++)
        for(int q=0; q<num_dims; q++)
         for(int r=0; r<num_dims; r++)
          for(int s=0; s<num_dims; s++)
           for(int t=0; t<num_dims; t++)
            inGlobalBasis(i,j,k,l)
              += inMatlBasis(q,r,s,t)*matlBasis(i,q)*matlBasis(j,r)
                                     *matlBasis(k,s)*matlBasis(l,t);
}
/******************************************************************************/
template<typename DataT>
void
FM::changeBasis(       Intrepid2::Tensor3<DataT, FM::FM_3D>& inGlobalBasis,
                 const Intrepid2::Tensor3<DataT, FM::FM_3D>& inMatlBasis,
                 const Intrepid2::Tensor <DataT, FM::FM_3D>& matlBasis)
/******************************************************************************/
{
    int num_dims = matlBasis.get_dimension();
    inGlobalBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int k=0; k<num_dims; k++)
       for(int q=0; q<num_dims; q++)
        for(int r=0; r<num_dims; r++)
         for(int s=0; s<num_dims; s++)
           inGlobalBasis(i,j,k)
             += inMatlBasis(q,r,s)*matlBasis(i,q)*matlBasis(j,r)*matlBasis(k,s);
}
/******************************************************************************/
template<typename DataT>
void
FM::changeBasis(       Intrepid2::Tensor<DataT, FM::FM_3D>& inGlobalBasis,
                 const Intrepid2::Tensor<DataT, FM::FM_3D>& inMatlBasis,
                 const Intrepid2::Tensor<DataT, FM::FM_3D>& matlBasis)
/******************************************************************************/
{
    int num_dims = matlBasis.get_dimension();
    inGlobalBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int j=0; j<num_dims; j++)
      for(int q=0; q<num_dims; q++)
       for(int r=0; r<num_dims; r++)
        inGlobalBasis(i,j) += inMatlBasis(q,r)*matlBasis(i,q)*matlBasis(j,r);
}
/******************************************************************************/
template<typename DataT>
void
FM::changeBasis(       Intrepid2::Vector<DataT, FM::FM_3D>& inGlobalBasis,
                 const Intrepid2::Vector<DataT, FM::FM_3D>& inMatlBasis,
                 const Intrepid2::Tensor<DataT, FM::FM_3D>& matlBasis)
/******************************************************************************/
{
    int num_dims = matlBasis.get_dimension();
    inGlobalBasis.clear();
    for(int i=0; i<num_dims; i++)
     for(int q=0; q<num_dims; q++)
      inGlobalBasis(i) += inMatlBasis(q)*matlBasis(i,q);
}


/******************************************************************************/
template<typename NLS, typename ArgT>
void
FM::DescentNorm(NLS & nls, Intrepid2::Vector<ArgT, FM::MAX_TRNS> & xi)
/******************************************************************************/
{
  if( xi.get_dimension() == 0 ) return;


  //length of xi (number of active transitions)
  int len_xi = xi.get_dimension();

  Intrepid2::Vector<ArgT, FM::MAX_TRNS> max_xi(len_xi);
  //Creation of different test xi
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> t_xi0(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> t_xi1(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> t_xi2(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> t_xi3(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> t_xi4(len_xi);


  Intrepid2::Vector<ArgT, FM::MAX_TRNS> residual0(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> residual1(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> residual2(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> residual3(len_xi);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> residual4(len_xi);

  Intrepid2::Vector<ArgT, FM::MAX_TRNS> maxR(5);

  Intrepid2::Vector<ArgT, FM::MAX_TRNS> invA0(5);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> invA1(5);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> invA2(5);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> invA3(5);
  Intrepid2::Vector<ArgT, FM::MAX_TRNS> invA4(5);
  invA0[0] =    1;
  invA0[1] =    0;
  invA0[2] =    0;
  invA0[3] =    0;
  invA0[4] =    0;
  invA1[0] =  -15;
  invA1[1] =   24.380952380952380;
  invA1[2] =  -10.666666666666667;
  invA1[3] =    1.333333333333333;
  invA1[4] =   -0.047619047619048;
  invA2[0] =   70;
  invA2[1] = -170.6666666666666667;
  invA2[2] =  117.333333333333333;
  invA2[3] =  -17.333333333333333;
  invA2[4] =    0.666666666666667;
  invA3[0] = -120;
  invA3[1] =  341.333333333333333;
  invA3[2] = -277.333333333333333;
  invA3[3] =   58.666666666666667;
  invA3[4] =   -2.666666666666667;
  invA4[0] =   64;
  invA4[1] = -195.047619047619048;
  invA4[2] =  170.666666666666667;
  invA4[3] =  -42.666666666666667;
  invA4[4] =    3.047619047619048;

  Intrepid2::Vector<ArgT, FM::MAX_TRNS> C_fx(5);

  ArgT xmin;
  ArgT xmax;
  ArgT xroot;

  ArgT residmin;
  ArgT residmax;
  ArgT residroot;


  for (int nState = 0; nState<2; nState++){


    //updateXi calculates the maximum Xi step (if T and E are held fixed)
    max_xi = nls.updateXi(xi);

    t_xi0 = xi + 0*max_xi;
    t_xi1 = xi + 0.125*max_xi;
    t_xi2 = xi + 0.25*max_xi;
    t_xi3 = xi + 0.5*max_xi;
    t_xi4 = xi + 1*max_xi;

    residual0 = nls.conditionedGrad(t_xi0,xi);
    residual1 = nls.conditionedGrad(t_xi1,xi);
    residual2 = nls.conditionedGrad(t_xi2,xi);
    residual3 = nls.conditionedGrad(t_xi3,xi);
    residual4 = nls.conditionedGrad(t_xi4,xi);

     maxR.clear();

    for (int i=0; i<len_xi; i++) {
      if (residual0[i]<maxR[0]) {
        maxR[0] = residual0[i];
        maxR[1] = residual1[i];
        maxR[2] = residual2[i];
        maxR[3] = residual3[i];
        maxR[4] = residual4[i];
      }
    }

    //COEFFICIENTS for maxR = C0 + C1*x + C2*x^2 + C3*x^3 + C4*x^4
    //C = invA*maxR
    C_fx.clear();

    for (int i=0; i<5; i++) {
      C_fx[0] += invA0[i]*maxR[i];
      C_fx[1] += invA1[i]*maxR[i];
      C_fx[2] += invA2[i]*maxR[i];
      C_fx[3] += invA3[i]*maxR[i];
      C_fx[4] += invA4[i]*maxR[i];
    }

    //BISECTION ROOT METHOD FOR FINDING ROOT OF
    //R = 0 = C0 + C1*x + C2*x^2 + C3*x^3 + C4*x^4
    xmin = 0;
    xmax = 1;
    xroot = 0;

    residmin = maxR[0];
    residmax = maxR[4];
    residroot = maxR[0];
    int w_iter = 0;

    while ((abs(residroot) > 100) && (w_iter < 100)) {
      w_iter++;
      //weighted bisection method
      xroot = xmin*0.7 + xmax*0.3;
      residroot = C_fx[0] + C_fx[1]*xroot + C_fx[2]*xroot*xroot + C_fx[3]*xroot*xroot*xroot + C_fx[4]*xroot*xroot*xroot*xroot;

      // if residroot and residmin have the same sign and the residual is not growing
      if ((residroot*residmin>=0)&&(abs(residmin)>=abs(residroot))) {
        xmin = xroot;
        residmin = residroot;
      }
      else {
        xmax = xroot;
        residmax = residroot;
      }
    }  //END WHILE

    xi += xroot*max_xi;
  } //END FOR nState

}


/******************************************************************************/
template<typename NLS, typename ArgT>
void
FM::ScaledDescent(NLS & nls, Intrepid2::Vector<ArgT, FM::MAX_TRNS> & xi)
/******************************************************************************/
{

  if( xi.get_dimension() == 0 ) return;


  Intrepid2::Vector<ArgT, FM::MAX_TRNS>
  residual = nls.gradient(xi);

  int iter = 0;
  ArgT resnorm = Intrepid2::norm(residual);

  while(resnorm > 1e4 && iter < 10){

    Intrepid2::Tensor<ArgT, FM::MAX_TRNS>
      hessian = nls.hessian(xi);

    Intrepid2::Vector<ArgT, FM::MAX_TRNS>
    diag = Intrepid2::diag(hessian);

    ArgT diag1norm = Intrepid2::norm_1(diag);

    // diag1norm /= xi.get_dimension();

    xi -= residual / diag1norm;

    residual = nls.gradient(xi);
    resnorm = Intrepid2::norm(residual);
    iter++;

  }
}


/******************************************************************************/
template<typename DataT, typename ArgT>
void
FM::computeBinFractions(
    Intrepid2::Vector<ArgT, FM::MAX_TRNS> const & xi,
    Teuchos::Array<ArgT>                        & newFractions,
    Teuchos::Array<DataT>                 const & oldFractions,
    Teuchos::Array<int>                   const & transitionMap,
    Intrepid::FieldContainer<DataT>       const & aMatrix)
/******************************************************************************/
{
  int nVariants = oldFractions.size();
  int nTransitions = transitionMap.size();
  for(int I=0;I<nVariants;I++){
    newFractions[I] = oldFractions[I];
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        newFractions[I] += xi(transitionMap[i])*aMatrix(I,i);
      }
    }
  }
}

/******************************************************************************/
template<typename ArgT>
void
FM::computeInitialState(
    Teuchos::Array<RealType>            const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::FM_3D> const & x,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & X,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::FM_3D> const & E,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & D,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & linear_D)
/******************************************************************************/
{
  Intrepid2::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  Intrepid2::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  Intrepid2::Tensor <ArgT,FM::FM_3D> k; k.clear();

  Intrepid2::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  Intrepid2::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();

  int nVariants = crystalVariants.size();
  for(int i=0; i<nVariants; i++){
    const CrystalVariant& variant = crystalVariants[i];
    remanent_x += fractions[i]*variant.spontStrain;
    remanent_D += fractions[i]*variant.spontEDisp;
    C += fractions[i]*variant.C;
    ep += fractions[i]*variant.ep;
    k += fractions[i]*variant.k;
  }

  linear_x = x - remanent_x;
  X = dotdot(C,linear_x) - dot(E,ep);


  linear_D = dotdot(ep, linear_x) + dot(k, E);
  D = linear_D + remanent_D;
}

/******************************************************************************/
template<typename ArgT>
void
FM::computeRelaxedState(
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants,
    Intrepid2::Tensor<ArgT,FM::FM_3D> const & x,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & X,
    Intrepid2::Tensor<ArgT,FM::FM_3D>       & linear_x,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & E,
    Intrepid2::Vector<ArgT,FM::FM_3D> const & D,
    Intrepid2::Vector<ArgT,FM::FM_3D>       & linear_D)
/******************************************************************************/
{
  Intrepid2::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  Intrepid2::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  Intrepid2::Tensor <ArgT,FM::FM_3D> k; k.clear();

  Intrepid2::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  Intrepid2::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();

  int nVariants = crystalVariants.size();
  for(int i=0; i<nVariants; i++){
    const CrystalVariant& variant = crystalVariants[i];
    remanent_x += fractions[i]*variant.spontStrain;
    remanent_D += fractions[i]*variant.spontEDisp;
    C += fractions[i]*variant.C;
    ep += fractions[i]*variant.ep;
    k += fractions[i]*variant.k;
  }

  linear_x = x - remanent_x;
  linear_D = D - remanent_D;

  Intrepid2::Tensor<ArgT,FM::FM_3D> b = Intrepid2::inverse(k);

  E = dot(b, (linear_D - dotdot(ep, linear_x)));
  X = dotdot(C,linear_x) - dot(E,ep);


}

/******************************************************************************/
template<typename DataT, typename ArgT>
void
FM::computeResidual(
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
    Intrepid2::Vector<ArgT,FM::FM_3D>   const & linear_D)
/******************************************************************************/
{
  int nVariants = fractions.size();
  ArgT half = 1.0/2.0;
  for(int I=0;I<nVariants;I++){
    ArgT fracI = fractions[I];
    for(int J=0;J<nVariants;J++){
      int i=I*nVariants+J;
      if(transitionMap[i] >= 0){
        const Transition& transition = transitions[i];
        int lindex = transitionMap[i];
        residual[lindex] = -tBarrier[i]
                           -dotdot(transition.transStrain, X)
                           -dot(transition.transEDisp, E);
      }
    }
  }
/*
  int nTransitions = transitions.size();
  for(int I=0;I<nVariants;I++){
    ArgT myRate(0.0);
    const CrystalVariant& variant = crystalVariants[I];
    myRate += dotdot(linear_x,dotdot(variant.C,linear_x)-dot(linear_D,variant.h))*half;
    myRate += dot(linear_D,dot(variant.b,linear_D)-dotdot(variant.h,linear_x))*half;
    for(int i=0;i<nTransitions;i++){
      if(transitionMap[i] >= 0){
        DataT aVal = aMatrix(I,i);
        if( aVal == 0.0 ) continue;
        int lindex = transitionMap[i];
        residual[lindex] += aVal*myRate;
      }
    }
  }
  */
}

/******************************************************************************/
template<typename ArgT>
void
FM::computePermittivity(
    Intrepid2::Tensor<ArgT,FM::FM_3D>         & eps,
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants)
/******************************************************************************/
{
  Intrepid2::Tensor <ArgT,FM::FM_3D> k; k.clear();
  int nVariants = crystalVariants.size();
  for(int i=0; i<nVariants; i++){
    const CrystalVariant& variant = crystalVariants[i];
    k += fractions[i]*variant.k;
  }
  eps = k;
}

/******************************************************************************/
template<typename ArgT>
void
FM::computePolarization(
    Intrepid2::Vector<ArgT,FM::FM_3D>         & pol,
    Teuchos::Array<ArgT>                const & fractions,
    Teuchos::Array<FM::CrystalVariant>  const & crystalVariants)
/******************************************************************************/
{
  pol.clear();
  int nVariants = crystalVariants.size();
  for(int i=0; i<nVariants; i++){
    const CrystalVariant& variant = crystalVariants[i];
    pol += fractions[i]*variant.spontEDisp;
  }
}



/******************************************************************************/
template<typename ArgT>
void
FM::computeMaxCD(
    Teuchos::Array<ArgT>                    & max_CD,
    Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >   const & crystalPhases)
/******************************************************************************/
{
  Intrepid2::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  Intrepid2::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  Intrepid2::Tensor <ArgT,FM::FM_3D> k; k.clear();
  int nPhases = crystalPhases.size();

  ArgT C_D_11;
  ArgT C_D_22;
  ArgT C_D_33;

  max_CD[0] = 0;
  for(int i=0; i<nPhases; i++){
    C = crystalPhases[i]->C;
    ep = crystalPhases[i]->ep;
    k = crystalPhases[i]->k;

    C_D_11 = C(0,0,0,0) + ep(2,0,0)*ep(2,0,0)/k(2,2);
    C_D_22 = C(1,1,1,1) + ep(2,1,1)*ep(2,1,1)/k(2,2);
    C_D_33 = C(2,2,2,2) + ep(2,2,2)*ep(2,2,2)/k(2,2);

  }

  if (max_CD[0]<C_D_11) {
    max_CD[0] = C_D_11;
  }
  if (max_CD[0]<C_D_22) {
    max_CD[0] = C_D_22;
  }
  if (max_CD[0]<C_D_33) {
    max_CD[0] = C_D_33;
  }

}
