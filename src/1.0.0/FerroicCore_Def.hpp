//*****************************************************************//
//    Ferroic Model (gnom3-fe) 1.0.0:                              //
//    Copyright 2017 Sandia Corporation                            //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level gnom3-fe directory//
//*****************************************************************//
//


/******************************************************************************/
template<typename EvalT>
FM::DomainSwitching<EvalT>::DomainSwitching(
      Teuchos::Array<FM::CrystalGrain>    const & crystalGrains,
      Teuchos::Array<RealType>            const & transBarriers,
      Teuchos::Array<RealType>            const & deltaBarriers,
      Teuchos::Array<RealType>            const & polynomialN,
      Teuchos::Array<RealType>            const & binFractions,
      minitensor::Tensor<ArgT,FM::FM_3D>       const & x,
      minitensor::Vector<ArgT,FM::FM_3D>       const & E,
      RealType                                    dt,
      Teuchos::Array<int>                 const & nVals)
  :
      m_crystalGrains(crystalGrains),
      //m_transitions(transitions),
      m_transBarriers(transBarriers),
      m_deltaBarriers(deltaBarriers),
      m_polynomialN(polynomialN),
      m_binFractions(binFractions),
      //m_aMatrix(aMatrix),
      m_x(x), m_E(E), m_dt(dt), m_nVals(nVals)
/******************************************************************************/
{

  // compute trial state
  //
  minitensor::Tensor<ArgT, FM::FM_3D> X;
  minitensor::Tensor<ArgT, FM::FM_3D> linear_x;
  minitensor::Vector<ArgT, FM::FM_3D> linear_D;
  
  
  //THIS SETS m_D, 
  FM::computeInitialState(m_binFractions,      //const
                          m_crystalGrains,     //const
                          m_x,                 //const
             /*O*/        X,                   //output
             /*O*/        linear_x,            //output
                          m_E,                 //const
             /*O*/        m_D,                 //output
             /*O*/        linear_D,            //output 
                          m_nVals);            //const


  // set all transitions active for first residual eval
  //
  
  //int ngrains = m_nVals[0];
  //int nvars = m_nVals[1];
  //int ntrans = ngrains*nvars;
  //int ngrainstrans = ngrains*ntrans;
  //
  //
  //
  

  // evaluate residual at current bin fractions
  //
  
  //IS THIS SECTION NEEDED???
  //minitensor::Vector<ArgT, FM::MAX_GTRN> zero;
  //zero.set_dimension(ngrainstrans);
  //zero.clear();
  //minitensor::Vector<ArgT, FM::MAX_GTRN> residual = this->gradient(zero);

  
  
}

/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
T
FM::DomainSwitching<EvalT>::value(minitensor::Vector<T, N> const & x)
/******************************************************************************/
{
  return minitensor::Function_Base<
    DomainSwitching<EvalT>, typename EvalT::ScalarT, FM::MAX_GRNS>::value(*this, x);
}

/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
FM::DomainSwitching<EvalT>::gradient(
  minitensor::Vector<T, N> const & xi) const
/******************************************************************************/
{

  minitensor::Tensor<T, FM::FM_3D> X;         X.clear();
  minitensor::Tensor<T, FM::FM_3D> linear_x;  linear_x.clear();

  minitensor::Vector<T, FM::FM_3D> E;         E.clear();
  minitensor::Vector<T, FM::FM_3D> linear_D;  linear_D.clear();
  
  
  minitensor::Tensor<T, FM::FM_3D> const x_peeled = LCM::peel_tensor<EvalT, T, N, FM::FM_3D>()(m_x);

  minitensor::Vector<T, FM::FM_3D> const D_peeled = LCM::peel_vector<EvalT, T, N, FM::FM_3D>()(m_D);
  
  
  // apply transition increment (effect of xi on bin fractions)
  //
  int ngrains = m_nVals[0];
  int nvars = m_nVals[1];
  int ntrans = nvars*nvars;
  int ngrainsvars = ngrains*nvars;
  int ngrainstrans = ngrains*ntrans;

  //volume fractions for all grains-variants (material)
  Teuchos::Array<T> fractionsNew(ngrainsvars);   //Returned
    
 
  //old and new volumefractions for each grain
  Teuchos::Array<T> g_fractionsNew(nvars);
  Teuchos::Array<T> g_binFractions(nvars);   
  minitensor::Vector<T, FM::MAX_TRNS> g_xi(ntrans);
  
  for (int h = 0; h<ngrains; h++) {
        
    
        
    const FM::CrystalGrain& cg = m_crystalGrains[h];
        
    g_xi.clear();
    //PARSE xi (material) to g_xi (grain specific)    
    for (int i = 0; i<ntrans; i++) {
      g_xi[i] = xi[h*ntrans+i];
    }
    
    
    g_binFractions.clear();     //parse from input
    //PARSE m_binFractions (material) to g_binFractions (grain specific)    
    for (int i = 0; i<nvars; i++) {
      g_binFractions[i] = m_binFractions[h*nvars+i];
    } 
    g_fractionsNew.clear();     //parse to output
    
    
    //COMPUTE CHANGE TO g_fractionsNew  
    computeBinFractions(g_xi,               //const
                        g_fractionsNew, 
                        g_binFractions,     //const
                        cg.aMatrix,         //const
                        m_nVals);           //const
    
    //PARSE g_fractions (grain) new to fractionsNew (material)
    for (int i = 0; i<nvars; i++) {
      fractionsNew[h*nvars+i] = g_fractionsNew[i];
    }                
  }// END FOR h




  
  
  // compute new state
  //
  
  computeRelaxedState(fractionsNew,         //const
                      m_crystalGrains,      //const
                      x_peeled,             //const
                      X,                    //out      
                      linear_x,             //out      
                      E,                    //out      
                      D_peeled,             //const
                      linear_D,             //out
                      m_nVals);             //const            

  // compute new residual
  //
  minitensor::Vector<T, N> residual(ngrainstrans);
  
  
  
  computeResidual(residual,               //output
                  m_crystalGrains,        //const
                  m_transBarriers,        //const
                  X,                      //const
                  m_E,                    //const
                  m_nVals);               //const
                 

  return residual;


}



/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
FM::DomainSwitching<EvalT>::gradientStiff(
  minitensor::Vector<T, N> const & xi) const
/******************************************************************************/
{

  minitensor::Tensor<T, FM::FM_3D> X;         X.clear();
  minitensor::Tensor<T, FM::FM_3D> linear_x;  linear_x.clear();

  minitensor::Vector<T, FM::FM_3D> D;         D.clear();
  minitensor::Vector<T, FM::FM_3D> linear_D;  linear_D.clear();

  // apply transition increment (effect of xi on bin fractions)
  //
  int ngrains = m_nVals[0];
  int nvars = m_nVals[1];
  int ntrans = nvars*nvars;
  int ngrainsvars = ngrains*nvars;
  int ngrainstrans = ngrains*ntrans;

  //volume fractions for all grains-variants (material)
  Teuchos::Array<T> fractionsNew(ngrainsvars);   //Returned
    
  //old and new volumefractions for each grain
  Teuchos::Array<T> g_fractionsNew(nvars);
  Teuchos::Array<T> g_binFractions(nvars);   

  minitensor::Vector<T, FM::MAX_TRNS> g_xi;
  g_xi.set_dimension(ntrans); 
  
  for (int h = 0; h<ngrains; h++) {
        
    
        
    const FM::CrystalGrain& cg = m_crystalGrains[h];
        
    g_xi.clear();
    
    //PARSE xi (material) to g_xi (grain specific)    
    for (int i = 0; i<ntrans; i++) {
      g_xi[i] = xi[h*ntrans+i];
    }
    
    
    g_binFractions.clear();     //parse from input
    //PARSE m_binFractions (material) to g_binFractions (grain specific)    
    for (int i = 0; i<nvars; i++) {
      g_binFractions[i] = m_binFractions[h*nvars+i];
    } 
    g_fractionsNew.clear();     //parse to output
    
    
    //COMPUTE CHANGE TO g_fractionsNew 
    computeBinFractions(g_xi,               //const
                        g_fractionsNew, 
                        g_binFractions,     //const
                        cg.aMatrix,         //const
                        m_nVals);           //const 
    
    
    //PARSE g_fractions (grain) new to fractionsNew (material)
    for (int i = 0; i<nvars; i++) {
      fractionsNew[h*nvars+i] = g_fractionsNew[i];
    }                
  }// END FOR h
  
  
  

  // compute new state
  //
  FM::computeFinalState(fractionsNew,     //const
                      m_crystalGrains,    //const
                      m_x,                //const
                      X,                  //out
                      linear_x,           //out
                      m_E,                //const
                      D,                  //out
                      linear_D,           //out
                      m_nVals);           //const      




  // compute new residual
  //
  //
  minitensor::Vector<T, N> residual(ngrainstrans);
  
  
  
  computeResidual(residual,               //output
                  m_crystalGrains,        //const
                  m_transBarriers,        //const
                  X,                      //const
                  m_E,                    //const
                  m_nVals);               //const
                 

  return residual;
}




/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Tensor<T, N>
FM::DomainSwitching<EvalT>::hessian(
    minitensor::Vector<T, N> const & xi)
/******************************************************************************/
{
//  return minitensor::Function_Base<DomainSwitching<EvalT>,ArgT>::hessian(*this,xi);

  using AD = Sacado::Fad::SLFad<T, N>;

  minitensor::Index const
  dimension = xi.get_dimension();

  minitensor::Vector<AD, N>
  x_ad(dimension);

  for (minitensor::Index i{0}; i < dimension; ++i) {
    x_ad(i) = AD(dimension, i, xi(i));
  }

  minitensor::Vector<AD, N> const
  r_ad = this->gradient(x_ad);

  minitensor::Tensor<T, N>
  Hessian(dimension);

  for (minitensor::Index i{0}; i < dimension; ++i) {
    for (minitensor::Index j{0}; j < dimension; ++j) {
      Hessian(i, j) = r_ad(i).dx(j);
    }
  }

  return Hessian;
}



/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, FM::MAX_GVTS>
FM::DomainSwitching<EvalT>::computeVolFrac(
    minitensor::Vector<T, N> const & xi)
/******************************************************************************/
{
    
    //This function returns volumefractions after modifications by Xi
    
    int ngrains = m_nVals[0];
    int nvars = m_nVals[1];
    int ntrans = nvars*nvars;
    int ngrainsvars = ngrains*nvars;
    //int ngrainstrans = ngrains*ntrans;      //not used in this function

    
    Teuchos::Array<T> g_fractionsNew(nvars);
    Teuchos::Array<T> g_binFractions(nvars); 
    
    minitensor::Vector<T, FM::MAX_GVTS> fractions;
    fractions.set_dimension(ngrainsvars);
    fractions.clear();//Returned
    
    minitensor::Vector<T, FM::MAX_TRNS> g_xi;
    g_xi.set_dimension(ntrans);
    
    for (int h = 0; h<ngrains; h++) {
        
        g_fractionsNew.clear();     //parse to output
        g_binFractions.clear();     //parse from input
        
        const FM::CrystalGrain& cg = m_crystalGrains[h];
        
        
        g_xi.clear();
        
        for (int i = 0; i<ntrans; i++) {
            g_xi[i] = xi[h*ntrans+i];
        }
        
        
        
        for (int i = 0; i<nvars; i++) {
            g_binFractions[i] = m_binFractions[h*nvars+i];
        } 
        
         
        
        computeBinFractions(g_xi,               //const
                        g_fractionsNew, 
                        g_binFractions,     //const
                        cg.aMatrix,         //const
                        m_nVals);           //const
            
        
        
        for (int i = 0; i<nvars; i++) {
            fractions[h*nvars+i] = g_binFractions[i];
            
        }                
    }
    
    return fractions;
    
    
}


/******************************************************************************/
template<typename EvalT>
template<typename T, minitensor::Index N>
minitensor::Vector<T, N>
FM::DomainSwitching<EvalT>::phiNew(
    minitensor::Vector<T, N> const & xi)
/******************************************************************************/
{
    
    //This function returns volumefractions after modifications by Xi
    
    int ngrains = m_nVals[0];
    int nvars = m_nVals[1];
    int ntrans = nvars*nvars;
    int ngrainstrans = ngrains*ntrans;
    
    
    //auto const ngrainstrans = xi.get_dimension();
    
    minitensor::Vector<T, N> phi_new(ngrainstrans); phi_new.clear();
    
    
    
    minitensor::Vector<T, N> grad = FM::DomainSwitching<EvalT>::gradientStiff(xi);
    
    minitensor::Vector<T, FM::MAX_TRNS> g_grad;
    g_grad.set_dimension(ntrans); 
    g_grad.clear();
    
    minitensor::Vector<T, FM::MAX_TRNS> g_phi_new;
    g_phi_new.set_dimension(ntrans); g_phi_new.clear();
    
    for (int h = 0; h<ngrains; h++) {
        g_grad.clear();
        g_phi_new.clear();
        
        for (int i = 0; i<ntrans; i++) {
            g_grad[i] = grad[h*ntrans+i];
        }
        
        ES_computePhiNew(g_phi_new,         //output
                         g_grad,            //input
                         m_deltaBarriers,   //const
                         m_polynomialN,     //const
                         ntrans);           //const
        
        for (int i = 0; i<ntrans; i++) {
            phi_new[h*ntrans+i] = g_phi_new[i];
        }
        
    }  
    return phi_new;
    
}


/******************************************************************************/
template<typename DataT>
void
FM::changeBasis(       minitensor::Tensor4<DataT, FM::FM_3D>& inGlobalBasis,
                 const minitensor::Tensor4<DataT, FM::FM_3D>& inMatlBasis,
                 const minitensor::Tensor <DataT, FM::FM_3D>& matlBasis)
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
FM::changeBasis(       minitensor::Tensor3<DataT, FM::FM_3D>& inGlobalBasis,
                 const minitensor::Tensor3<DataT, FM::FM_3D>& inMatlBasis,
                 const minitensor::Tensor <DataT, FM::FM_3D>& matlBasis)
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
FM::changeBasis(       minitensor::Tensor<DataT, FM::FM_3D>& inGlobalBasis,
                 const minitensor::Tensor<DataT, FM::FM_3D>& inMatlBasis,
                 const minitensor::Tensor<DataT, FM::FM_3D>& matlBasis)
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
FM::changeBasis(       minitensor::Vector<DataT, FM::FM_3D>& inGlobalBasis,
                 const minitensor::Vector<DataT, FM::FM_3D>& inMatlBasis,
                 const minitensor::Tensor<DataT, FM::FM_3D>& matlBasis)
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
FM::Explicit_Smooth(NLS & nls, minitensor::Vector<ArgT, FM::MAX_GTRN> & xi)
/******************************************************************************/
{
    if (xi.get_dimension() == 0) return;
    
    
    
    // Initialization of scalars;
    
        
    const Teuchos::Array<int>& nVals = nls.get_nVals();
    int ngrains = nVals[0];
    int nvars = nVals[1];
    int ntrans = nvars*nvars;
    int ngrainsvars = ngrains*nvars;
    int ngrainstrans = ngrains*ntrans;    
          
            
    //Material Variables 
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> grad;
    grad.set_dimension(ngrainstrans); 
    grad.clear();       //defined
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_phi_new;
    ES_phi_new.set_dimension(ngrainstrans); 
    ES_phi_new.clear();  
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_phi_old;
    ES_phi_old.set_dimension(ngrainstrans);
    ES_phi_old.clear();
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_delta_phi_0;
    ES_delta_phi_0.set_dimension(ngrainstrans); 
    ES_delta_phi_0.clear();
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_delta_phi_1;
    ES_delta_phi_1.set_dimension(ngrainstrans); 
    ES_delta_phi_1.clear();
    
    minitensor::Vector<ArgT, FM::MAX_GVTS> volFrac;
    volFrac.set_dimension(ngrainsvars); 
    volFrac.clear();    //defined
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> temp_xi;
    temp_xi.set_dimension(ngrainstrans);  
    temp_xi.clear();        
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> max_xi;
    max_xi.set_dimension(ngrainstrans); 
    max_xi.clear();  // xi vector if constant stress and efield
                        
    
    //Grain Specific Variables initialize Linear Algebra Matrices
    minitensor::Vector<ArgT, FM::MAX_TRNS> A;
    A.set_dimension(ntrans);
    A.clear();
    
    minitensor::Vector<ArgT, FM::MAX_VRNT> B;
    B.set_dimension(nvars);
    B.clear();
    
    minitensor::Vector<ArgT, FM::MAX_VRNT> X;
    X.set_dimension(nvars);
    X.clear();
    
    minitensor::Vector<ArgT, FM::MAX_TRNS> Aeye;
    Aeye.set_dimension(ntrans);    // Should be identity after solving AX=B
    Aeye.clear();                    
                                            
           
                    
    //scalars    
    int igrainstrans;
       
    ArgT vI;                    // volfrac of Ith variant
    ArgT vJ;                    // volfrac of Jth variant
    ArgT xi_factor;             // multiplier for temp_xi = xi + xi_factor*max_xi
    ArgT max_xi_elem;           // maximum element of temp_xi
        
    ArgT zero_tol = 1e-4;      //tolerance for numerical value of vI and phi to be considered zero.  
    ArgT step_tol = 1e-4;
    ArgT max_Phi = 0.999;
    
    int dc_ctr;   //decouple counter    
    int decouple_flg;           //flag for further decouple of transformations in max_xi calculation
    
    int reduce_factor_flg;  //reduce factor flag
    int w_rf_ctr;           //reduce factor counter
         
    //ArgT dummy;  //dummy variable for calculations
        
    int xi_ctr;
    int xi_ctr_max = 1;
    xi_ctr=0;
    
    int xi_compute_flg = 1;    //checks to see if max_xi is zero.
    
    ArgT Xnet;      //net X
    ArgT Xgross;    //contributing portions only
    ArgT Xscaler;
    
    

    
    
    
while((xi_ctr<xi_ctr_max)&&(xi_compute_flg == 1)){
    xi_ctr++;
    //COMPUTE STEP
    
    
        
    max_xi_elem = 0;
    max_xi.clear();
    

    
    
    volFrac = nls.computeVolFrac(xi);     //NGRAINSxNVARS
    
    
    //Computed from phi = fn(grad)
    ES_phi_new = nls.phiNew(xi);
    
    //Computed from phi = vJ / (vI + vJ)
    ES_computePhiOld(ES_phi_old, volFrac, nvars, ngrains);
        
    for (int h = 0; h < ngrainstrans; h++) {
        if (ES_phi_new[h] >= max_Phi) {
            ES_phi_new[h] = max_Phi;  //to keep things positive definite and not singular
        }
    }    
    
    ES_delta_phi_0 = ES_phi_new - ES_phi_old;    
        
        
    for (int h = 0; h<ngrains; h++) {   //PER GRAIN CALCULATION OF XI_MAX
    
        for (int i = 0; i < nvars; i++) {
            vI = volFrac[h*nvars+i];
            A.clear();
            B.clear();
            X.clear();
            Aeye.clear();
            Xnet = 0;
            Xgross = 0;
            Xscaler = 1;
            
            if (vI > zero_tol) {   //If there is any volume fraction I to transform
                
                for (int j = 0; j < nvars; j++){
                    
                    vJ = volFrac[j];
                    igrainstrans = h*ntrans + i*nvars+j;
                    
                    //construct A and B
                    if (i==j){
                        A[j*nvars+j] = 1.0;
                        B[j]=0;
                    
                    }
                    else if ((ES_delta_phi_0[igrainstrans] > zero_tol)&&(ES_phi_new[igrainstrans] > zero_tol)) {
                        
                        A[j*nvars+i] = -ES_phi_new[igrainstrans];
                        A[j*nvars+j] = 1 - ES_phi_new[igrainstrans];
                        A[i*nvars+j] = 1;
                        B[j] = ES_phi_new[igrainstrans]*vI + (ES_phi_new[igrainstrans]-1)*vJ;
                    } 
                    else {
                        // Decouples Transformation
                        A[j*nvars+i] = 0.0;
                        A[i*nvars+j] = 0.0;
                        A[j*nvars+j] = 1.0;
                        B[j]=0;
                    } // END if 
                    
                } // END FOR j    
                
                
                //solve for X in AX = B
                ES_solveCondLE(A,X,B,nvars,i,Aeye);
                
                //check for Decoupling                
                decouple_flg = 1;
                dc_ctr=0;
                while ((decouple_flg==1)&&(dc_ctr<10)){
                    dc_ctr++;
                    //Check for and decouple negative transformations
                    for (int j = 0; j < nvars; j++) {
                        if ((i!=j)&&(X(j) < 0)) {
                            A[j*nvars+i] = 0.0;
                            A[i*nvars+j] = 0.0;
                            A[j*nvars+j] = 1.0;
                            B[j] = 0.0;      
                        }
                    }
                    
                    //Resolve for X
                    //localSolver.solve();
                    ES_solveCondLE(A,X,B,nvars,i,Aeye);
                    
                    //continue decouple loop check condition
                    decouple_flg = 0;                    
                    for (int j = 0; j < nvars; j++) {
                        if ((i!=j)&&(X(j) < 0)) {               
                            decouple_flg = 1;
                        }
                    }
                    
                } // end while decouple
            
                
            } //end if VI
            
            //Write to max_xi
            
            for (int j = 0; j < nvars; j++) { 
                Xnet += X(j);
                if (i!=j){
                    Xgross += X(j);
                }       
            }
            
            if (Xgross>zero_tol){
                X = X*(-X(i)/Xgross);
            }
            if (vI < -X(i)){
                Xscaler = -vI/X(i);
            }
            
            
            for (int j = 0; j < nvars; j++) { 
                if (i != j) {
                    igrainstrans = h*ntrans + i*nvars+j;
                    max_xi[igrainstrans] = X(j)*Xscaler;
                    
                }       
            }
            
            
            
            
        } //END FOR i
    
    } //END FOR h
    
    
    
    for (int g = 0; g < ngrainstrans; g++) { 
            if (max_xi[g] < 0) {
                max_xi[g] = 0;
            }
            if (max_xi[g] > max_xi_elem) {
                max_xi_elem = max_xi[g];    
            }
            
            // ERROR CHECK: max_xi should never be NaN
            TEUCHOS_TEST_FOR_EXCEPTION((max_xi[g]*0 != 0), std::invalid_argument, ">>> ERROR (FerroicCore::Explicit_Smooth) Max Xi = NAN, Max Xi"<<max_xi_elem);
            // ERROR CHECK: max_xi should never be > 1
            TEUCHOS_TEST_FOR_EXCEPTION(((max_xi[g] - 1) > zero_tol), std::invalid_argument, ">>> ERROR (FerroicCore::Explicit_Smooth) Max Xi > 1, Max Xi = "<<max_xi_elem);
            
     }
    
     if (max_xi_elem < zero_tol) {
        xi_compute_flg = 0;
     }
     

    
    //*            
                
                                
     xi_factor = 1;
     reduce_factor_flg = 1;
     w_rf_ctr = 0;
        
     while ((reduce_factor_flg == 1)&&(xi_factor >= step_tol)) {
                
         xi_factor = pow(0.5,w_rf_ctr);   //factor decreases by 1 order of magnitude per loop
         temp_xi = xi + xi_factor*max_xi;
         w_rf_ctr++;
         
         
         
         volFrac = nls.computeVolFrac(temp_xi);     //NGRAINSxNVARS
    
    
         //Computed from phi = fn(grad)
         ES_phi_new = nls.phiNew(temp_xi);
    
         //Computed from phi = vJ / (vI + vJ)
         ES_computePhiOld(ES_phi_old, volFrac, nvars, ngrains);
        
         for (int g = 0; g < ngrainstrans; g++) {
             if (ES_phi_new[g] >= max_Phi) {
                 ES_phi_new[g] = max_Phi;  //to keep things positive definite and not singular
             }
         }    
    
         ES_delta_phi_1 = ES_phi_new - ES_phi_old;    
         
         reduce_factor_flg = 0;
            
         for (int g = 0; g < ngrainstrans; g++) {
             if ((ES_delta_phi_0[g] > 0) && (ES_delta_phi_1[g]<0)) {
                 reduce_factor_flg = 1;    //continue to reduce factors.
             }
         }
            
     } //end reduce factor while loop
        
     for (int g = 0; g < ngrainstrans; g++) { 
            TEUCHOS_TEST_FOR_EXCEPTION(((temp_xi[g] - 1) > zero_tol), std::invalid_argument, ">>> ERROR (FerroicCore::Explicit_Smooth) Xi > 1, Xi = "<<temp_xi[g]);
     }

     xi = temp_xi;
     
    //*/

    //xi = max_xi;
    
       
          
                
}//end while
 
    

} //end Explicit Smooth





/******************************************************************************/
template<typename NLS, typename ArgT>
void
FM::Explicit_Smooth_Poling(NLS & nls, minitensor::Vector<ArgT, FM::MAX_GTRN> & xi)
/******************************************************************************/
{
    if (xi.get_dimension() == 0) return;
    
    
    
    // Initialization of scalars;
    
        
    const Teuchos::Array<int>& nVals = nls.get_nVals();
    int ngrains = nVals[0];
    int nvars = nVals[1];
    int ntrans = nvars*nvars;
    int ngrainsvars = ngrains*nvars;
    int ngrainstrans = ngrains*ntrans;    
          
            
    //Material Variables 
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> grad;
    grad.set_dimension(ngrainstrans); 
    grad.clear();       //defined
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_phi_new;
    ES_phi_new.set_dimension(ngrainstrans); 
    ES_phi_new.clear();  
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_phi_old;
    ES_phi_old.set_dimension(ngrainstrans);
    ES_phi_old.clear();
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> ES_delta_phi_0;
    ES_delta_phi_0.set_dimension(ngrainstrans); 
    ES_delta_phi_0.clear();
    
    
    
    minitensor::Vector<ArgT, FM::MAX_GVTS> volFrac;
    volFrac.set_dimension(ngrainsvars); 
    volFrac.clear();    //defined
    
    
    minitensor::Vector<ArgT, FM::MAX_GTRN> max_xi;
    max_xi.set_dimension(ngrainstrans); 
    max_xi.clear();  // xi vector if constant stress and efield
                        
    
    //Grain Specific Variables initialize Linear Algebra Matrices
    minitensor::Vector<ArgT, FM::MAX_TRNS> A;
    A.set_dimension(ntrans);
    A.clear();
    
    minitensor::Vector<ArgT, FM::MAX_VRNT> B;
    B.set_dimension(nvars);
    B.clear();
    
    minitensor::Vector<ArgT, FM::MAX_VRNT> X;
    X.set_dimension(nvars);
    X.clear();
    
    minitensor::Vector<ArgT, FM::MAX_TRNS> Aeye;
    Aeye.set_dimension(ntrans);    // Should be identity after solving AX=B
    Aeye.clear();                    
                                            
           
                    
    //scalars    
    int igrainstrans;
       
    ArgT vI;                    // volfrac of Ith variant
    ArgT vJ;                    // volfrac of Jth variant
    //ArgT xi_factor;             // multiplier for temp_xi = xi + xi_factor*max_xi  //not used in this function
    ArgT max_xi_elem;           // maximum element of temp_xi
        
    ArgT zero_tol = 1e-4;      //tolerance for numerical value of vI and phi to be considered zero.  
    //ArgT step_tol = 1e-4;       //not used in this function
    ArgT max_Phi = 0.999;
    
    int dc_ctr;   //decouple counter    
    int decouple_flg;           //flag for further decouple of transformations in max_xi calculation
    
    
    
    //COMPUTE STEP
    
    max_xi_elem = 0;
    max_xi.clear();
    
    volFrac = nls.computeVolFrac(xi);     //NGRAINSxNVARS
        
    //Computed from phi = fn(grad)
    ES_phi_new = nls.phiNew(xi);
    
    //Computed from phi = vJ / (vI + vJ)
    ES_computePhiOld(ES_phi_old, volFrac, nvars, ngrains);
        
    for (int h = 0; h < ngrainstrans; h++) {
        if (ES_phi_new[h] >= max_Phi) {
            ES_phi_new[h] = max_Phi;  //to keep things positive definite and not singular
        }
    }    
    
    ES_delta_phi_0 = ES_phi_new - ES_phi_old;    
        
        
    for (int h = 0; h<ngrains; h++) {   //PER GRAIN CALCULATION OF XI_MAX
    
        for (int i = 0; i < nvars; i++) {
            vI = volFrac[h*nvars+i];
            A.clear();
            B.clear();
            X.clear();
            Aeye.clear();
            
            
            if (vI > zero_tol) {   //If there is any volume fraction I to transform
                
                for (int j = 0; j < nvars; j++){
                    
                    vJ = volFrac[j];
                    igrainstrans = h*ntrans + i*nvars+j;
                    
                    //construct A and B
                    if (ES_delta_phi_0[igrainstrans] > zero_tol) {
                        
                        A[j*nvars+i] = -ES_phi_new[igrainstrans];
                        A[i*nvars+j] = 1;
                        A[j*nvars+j] = 1 - ES_phi_new[igrainstrans];
                        B[j] = ES_phi_new[igrainstrans]*vI + (ES_phi_new[igrainstrans]-1)*vJ;
                    } 
                    else {
                        // Decouples Transformation
                        A[j*nvars+i] = 0.0;
                        A[i*nvars+j] = 0.0;
                        A[j*nvars+j] = 1.0;
                        B[j]=0;
                    } // END if 
                    
                } // END FOR j    
                
                
                //solve for X in AX = B
                ES_solveCondLE(A,X,B,nvars,i,Aeye);
                
                //check for Decoupling                
                decouple_flg = 1;
                dc_ctr=0;
                while ((decouple_flg==1)&&(dc_ctr<2)){
                    dc_ctr++;
                    //Check for and decouple negative transformations
                    for (int j = 0; j < nvars; j++) {
                        if ((i!=j)&&(X(j) < 0)) {
                            A[j*nvars+i] = 0.0;
                            A[i*nvars+j] = 0.0;
                            A[j*nvars+j] = 1.0;
                            B[j] = 0.0;      
                        }
                    }
                    
                    //Resolve for X
                    //localSolver.solve();
                    ES_solveCondLE(A,X,B,nvars,i,Aeye);
                    
                    //continue decouple loop check condition
                    decouple_flg = 0;                    
                    for (int j = 0; j < nvars; j++) {
                        if ((i!=j)&&(X(j) < 0)) {               
                            decouple_flg = 1;
                        }
                    }
                    
                } // end while decouple
            
                
            } //end if VI
            
            //Write to max_xi
            for (int j = 0; j < nvars; j++) { 
                if (i != j) {
                    igrainstrans = h*ntrans + i*nvars+j;
                    max_xi[igrainstrans] = X(j);
                    
                }       
            }
 
            
        } //END FOR i
    
    } //END FOR h
    
    
    
    for (int g = 0; g < ngrainstrans; g++) { 
            if (max_xi[g] < 0) {
                max_xi[g] = 0;
            }
            if (max_xi[g] > max_xi_elem) {
                max_xi_elem = max_xi[g];    
            }
            
            // ERROR CHECK: max_xi should never be NaN
            TEUCHOS_TEST_FOR_EXCEPTION((max_xi[g]*0 != 0), std::invalid_argument, ">>> ERROR (FerroicCore::Explicit_Smooth) Max Xi = NAN, Max Xi"<<max_xi_elem);
            // ERROR CHECK: max_xi should never be > 1
            TEUCHOS_TEST_FOR_EXCEPTION((max_xi[g] > 1), std::invalid_argument, ">>> ERROR (FerroicCore::Explicit_Smooth) Max Xi > 1, Max Xi = "<<max_xi_elem);
            
     }
    
    

     xi = max_xi;

 

} //end Explicit Smooth Poling





/******************************************************************************/
template<typename NLS, typename ArgT>
void
FM::DescentNorm(NLS & nls, minitensor::Vector<ArgT, FM::MAX_TRNS> & xi)
/******************************************************************************/
{
    
     if( xi.get_dimension() == 0 ) return;
   

}


/******************************************************************************/
template<typename NLS, typename ArgT>
void
FM::ScaledDescent(NLS & nls, minitensor::Vector<ArgT, FM::MAX_TRNS> & xi)
/******************************************************************************/
{

  if( xi.get_dimension() == 0 ) return;

//
//  minitensor::Vector<ArgT, FM::MAX_TRNS>
//  residual = nls.gradient(xi);
//
//  int iter = 0;
//  ArgT resnorm = minitensor::norm(residual);
//
//  while(resnorm > 1e4 && iter < 10){
//
//    minitensor::Tensor<ArgT, FM::MAX_TRNS>
//      hessian = nls.hessian(xi);
//
//    minitensor::Vector<ArgT, FM::MAX_TRNS>
//    diag = minitensor::diag(hessian);
//
//    ArgT diag1norm = minitensor::norm_1(diag);
//
//    // diag1norm /= xi.get_dimension();
//
//    xi -= residual / diag1norm;
//
//    residual = nls.gradient(xi);
//    resnorm = minitensor::norm(residual);
//    iter++;
//
//  }
}


/******************************************************************************/
template<typename ArgT>
void
FM::computeBinFractions(
    minitensor::Vector<ArgT, FM::MAX_TRNS>     const & xi,
    Teuchos::Array<ArgT>                            & newFractions,
    Teuchos::Array<ArgT>                      const & oldFractions,
    minitensor::Vector<RealType, FM::MAX_AMTL> const & aMatrix,
    Teuchos::Array<int>                       const & nVals)
/******************************************************************************/
{
  int nvars = nVals[1];
  int ntrans = nvars*nvars;
  for(int I=0;I<nvars;I++){
    newFractions[I] = oldFractions[I];
    for(int i=0;i<ntrans;i++){
      newFractions[I] += xi(i)*aMatrix(I*ntrans+i);
    }
  }
}




/******************************************************************************/
template<typename ArgT>
void
FM::computeInitialState(
    Teuchos::Array<RealType>            const & fractions,  // grains x variants
    Teuchos::Array<FM::CrystalGrain>    const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D>   const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>         & X,
    minitensor::Tensor<ArgT,FM::FM_3D>         & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D>   const & E,
    minitensor::Vector<ArgT,FM::FM_3D>         & D,
    minitensor::Vector<ArgT,FM::FM_3D>         & linear_D,
    Teuchos::Array<int>                 const & nVals)
/******************************************************************************/
{
  minitensor::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  minitensor::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> k; k.clear();

  minitensor::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  minitensor::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();


  int ngrains = nVals[0];
  int nvars = nVals[1];
  ArgT vfgrain = 1.0/ngrains;
  
  int igrainsvars;
  
  for(int h=0; h<ngrains; h++){
    const FM::CrystalGrain& cg = crystalGrains[h];
    for(int i=0; i<nvars; i++){
        
        igrainsvars = h*nvars+i;
        
        const FM::CrystalVariant& variant = cg.crystalVariants[i];
        
        remanent_x += fractions[igrainsvars]*variant.spontStrain*vfgrain;
        remanent_D += fractions[igrainsvars]*variant.spontEDisp*vfgrain;
        C += fractions[igrainsvars]*variant.C*vfgrain;
        ep += fractions[igrainsvars]*variant.ep*vfgrain;
        k += fractions[igrainsvars]*variant.k*vfgrain;
    }
  }
  
  
  
  linear_x = x - remanent_x;
  X = dotdot(C,linear_x) - dot(E,ep);


  linear_D = dotdot(ep, linear_x) + dot(k, E);
  D = linear_D + remanent_D;
}


/******************************************************************************/
template<typename ArgT>
void
FM::computeFinalState(
    Teuchos::Array<ArgT>                const & fractions,   //grains x variants
    Teuchos::Array<FM::CrystalGrain>    const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D>   const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>         & X,
    minitensor::Tensor<ArgT,FM::FM_3D>         & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D>   const & E,
    minitensor::Vector<ArgT,FM::FM_3D>         & D,
    minitensor::Vector<ArgT,FM::FM_3D>         & linear_D,
    Teuchos::Array<int>                 const & nVals)
/******************************************************************************/
{
    
  //This function is exactly the same as computeInitialState however the datatype of <fractions> is different 
   
  minitensor::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  minitensor::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> k; k.clear();

  minitensor::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  minitensor::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();

  int ngrains = nVals[0];
  int nvars = nVals[1];
  ArgT vfgrain = 1.0/ngrains;
  
  int igrainsvars;
  
  for (int h=0; h<ngrains; h++){
    const FM::CrystalGrain& cg = crystalGrains[h]; 
     
    for(int i=0; i<nvars; i++){
        
        igrainsvars = h*nvars+i;
        
        const FM::CrystalVariant& variant = cg.crystalVariants[i];
        
        remanent_x += fractions[igrainsvars]*variant.spontStrain*vfgrain;
        remanent_D += fractions[igrainsvars]*variant.spontEDisp*vfgrain;
        C += fractions[igrainsvars]*variant.C*vfgrain;
        ep += fractions[igrainsvars]*variant.ep*vfgrain;
        k += fractions[igrainsvars]*variant.k*vfgrain;
    }
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
    Teuchos::Array<ArgT>              const & fractions,
    Teuchos::Array<FM::CrystalGrain>  const & crystalGrains,
    minitensor::Tensor<ArgT,FM::FM_3D> const & x,
    minitensor::Tensor<ArgT,FM::FM_3D>       & X,
    minitensor::Tensor<ArgT,FM::FM_3D>       & linear_x,
    minitensor::Vector<ArgT,FM::FM_3D>       & E,
    minitensor::Vector<ArgT,FM::FM_3D> const & D,
    minitensor::Vector<ArgT,FM::FM_3D>       & linear_D,
    Teuchos::Array<int>               const & nVals)
/******************************************************************************/
{
  minitensor::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  minitensor::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> k_var; k_var.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> b_var; b_var.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> b; b.clear();

  minitensor::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  minitensor::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();

  int ngrains = nVals[0];
  int nvars = nVals[1];
  ArgT vfgrain = 1.0/ngrains;
  
  int igrainsvars;
  
  for (int h=0; h<ngrains; h++){
    const FM::CrystalGrain& cg = crystalGrains[h];  
    for(int i=0; i<nvars; i++){
        
        igrainsvars = h*nvars+i;
        
        const FM::CrystalVariant& variant = cg.crystalVariants[i];
        
        remanent_x += fractions[igrainsvars]*variant.spontStrain*vfgrain;
        remanent_D += fractions[igrainsvars]*variant.spontEDisp*vfgrain;
        C += fractions[igrainsvars]*variant.C*vfgrain;
        ep += fractions[igrainsvars]*variant.ep*vfgrain;
        
        
        k_var.clear();
        b_var.clear();
        
        k_var += variant.ep;
        b_var = minitensor::inverse(k_var);
        
        b += fractions[igrainsvars]*b_var*vfgrain;
    }
  }
  

  linear_x = x - remanent_x;
  linear_D = D - remanent_D;

  E = dot(b, (linear_D - dotdot(ep, linear_x)));
  X = dotdot(C,linear_x) - dot(E,ep);


}

/******************************************************************************/
template<typename DataT, typename ArgT>
void
FM::computeResidual(
    minitensor::Vector<ArgT, FM::MAX_GTRN>       & residual,   //ngrains*ntrans
    Teuchos::Array<FM::CrystalGrain>      const & crystalGrains,
    Teuchos::Array<DataT>                 const & tBarrier,
    minitensor::Tensor<ArgT,FM::FM_3D>     const & X,
    minitensor::Vector<ArgT,FM::FM_3D>     const & E,
    Teuchos::Array<int>                   const & nVals)
/******************************************************************************/
{
  
  int ngrains = nVals[0];
  int nvars = nVals[1];
  
  //ArgT half = 1.0/2.0;     //not used in this function
  
  for (int H=0; H<ngrains; H++){
    
    const FM::CrystalGrain& cg = crystalGrains[H];
    
    for(int I=0;I<nvars;I++){
      
      for(int J=0;J<nvars;J++){
        int i=I*nvars+J;
        
        const Transition& transition = cg.transitions[i];
            
        residual[H*nvars*nvars+i] = -tBarrier[i]
                      -dotdot(transition.transStrain, X)
                      -dot(transition.transEDisp, E);
        
      }// END FOR J
    }// END FOR I
  }// END FOR H

}





/******************************************************************************/
template<typename ArgT>
void
FM::computeToAlegra(
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
    Teuchos::Array<int>                 const & nVals)
/******************************************************************************/
{
  
  //minitensor::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  minitensor::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> k; k.clear();
  
  minitensor::Tensor <ArgT,FM::FM_3D> remanent_x; remanent_x.clear();
  minitensor::Vector <ArgT,FM::FM_3D> remanent_D; remanent_D.clear();
  
  nd_pol.clear();
  eps.clear();
  vphases.clear();
  
  
  int ngrains = nVals[0];
  int nvars = nVals[1];
  ArgT vfgrain = 1.0/ngrains;
  
  int igrainsvars;
  
  ArgT DeltaFrac;
  ArgT maxDeltaFrac = 0;
  
  for (int h=0; h<ngrains; h++){
    const FM::CrystalGrain& cg = crystalGrains[h];
    
     
    for(int i=0; i<nvars; i++){
        
        igrainsvars = h*nvars + i;
        
        const FM::CrystalVariant& variant = cg.crystalVariants[i];
        
        remanent_x += fractions[igrainsvars]*variant.spontStrain*vfgrain;
        remanent_D += fractions[igrainsvars]*variant.spontEDisp*vfgrain;
                        
        //C += fractions[igrainsvars]*variant.C*vfgrain;
        ep += fractions[igrainsvars]*variant.ep*vfgrain;
                        
        k += fractions[igrainsvars]*variant.k*vfgrain;
        
        vphases[variant.phaseIndex] += fractions[igrainsvars]*vfgrain;
        
        
        DeltaFrac = fractions[igrainsvars] - oldfractions[igrainsvars];
        
        if (DeltaFrac > 0.0) {
            maxDeltaFrac += DeltaFrac*vfgrain;
        }
        
        
        
        
        
    } //end for i
    

    
  } //end for h
  
    
   
  /*
  
  P_nd = e * (S - S_r)
  
  */
  
  nd_pol = dotdot( ep, (x - remanent_x)) + remanent_D;
  
  eps = k;
  
  
  pressure[0] = -1.0/3.0*(X[0]+X[4]+X[8]);

  
  
  if (maxDeltaFrac > 1.0e-6) {
    PVFactor[0] = 2.0;
  } 
  else {
    PVFactor[0] = 0.0;
  }
  
  
  
  
  
}



/******************************************************************************/
template<typename ArgT>
void
FM::computeMaxCD(
    Teuchos::Array<ArgT>                                    & max_CD,
    Teuchos::Array<Teuchos::RCP<FM::CrystalPhase> >   const & crystalPhases,
    Teuchos::Array<int>                               const & nVals)
/******************************************************************************/
{
  minitensor::Tensor4<ArgT,FM::FM_3D> C; C.clear();
  minitensor::Tensor3<ArgT,FM::FM_3D> ep; ep.clear();
  minitensor::Tensor <ArgT,FM::FM_3D> k; k.clear();
  int nPhases = nVals[2];

  ArgT C_D_11=0.0;
  ArgT C_D_22=0.0;
  ArgT C_D_33=0.0;

  max_CD[0] = 0;
  for(int i=0; i<nPhases; i++){
    C = crystalPhases[i]->C;
    ep = crystalPhases[i]->ep;
    k = crystalPhases[i]->k;

    C_D_11 = C(0,0,0,0) + ep(2,0,0)*ep(2,0,0)/k(2,2);
    C_D_22 = C(1,1,1,1) + ep(2,1,1)*ep(2,1,1)/k(2,2);
    C_D_33 = C(2,2,2,2) + ep(2,2,2)*ep(2,2,2)/k(2,2);

  

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

}



/******************************************************************************/
template<typename ArgT>
void
FM::ES_computePhiNew(
    minitensor::Vector<ArgT, FM::MAX_TRNS>            & phi_new,
    minitensor::Vector<ArgT, FM::MAX_TRNS>      const & grad,
    Teuchos::Array<RealType>                   const & deltaG,
    Teuchos::Array<RealType>                   const & npower,
    int                                        const nTrans)
/******************************************************************************/
{
    //int nTrans = xi.get_dimension();
    ArgT x;
    
    ArgT n;
    ArgT Ncirc;
    ArgT iNcirc;

    ArgT dumb1;
    ArgT dumb2;
    ArgT dumb3;
    ArgT dumb4;

    for (int i = 0; i<nTrans; i++) {
        x = -grad[i]/deltaG[i];
        n = npower[i];
        Ncirc = 2*n;
        iNcirc = 1/Ncirc;
        
        dumb1=0;
        dumb2=0;
        dumb3=0;
        dumb4=0;
        
        if (x <= -1) 
        {
            phi_new[i] = 0;
        }
        else if (x <= 0)
        {
            dumb1 = x + 1.0;
            dumb2 = pow(dumb1,Ncirc);
            dumb3 = 1.0 - dumb2;
            dumb4 = pow(dumb3,iNcirc);
            
            phi_new[i] = 0.5*(1.0 - dumb4);
        }
        else if (x <= 1)
        {
            dumb1 = 1-x;
            dumb2 = pow(dumb1,Ncirc);
            dumb3 = 1.0 - dumb2;
            dumb4 = pow(dumb3,iNcirc);
            
            phi_new[i] = 0.5*(1.0 + dumb4);
        }
        else
        {
            phi_new[i] = 1;
        }   
    } //end For i
    
    
    
} //end computePhiNew


/******************************************************************************/
template<typename ArgT>
void
FM::ES_computePhiOld(
    minitensor::Vector<ArgT, FM::MAX_GTRN>            & phi_old,
    minitensor::Vector<ArgT, FM::MAX_GVTS>      const & oldVolFrac,
    int                                        const nvars,
    int                                        const ngrains)
/******************************************************************************/
{
    ArgT vI;
    ArgT vJ;
    

    
    int itr;
    
    for (int h = 0; h < ngrains; h++) {
    
        for (int i = 0; i < nvars; i++) {
            
            vI = oldVolFrac[h*nvars+i];
            
            
            for (int j = 0; j < nvars; j++) {
                
                vJ = oldVolFrac[h*nvars+j];
                
                itr = h*nvars*nvars + i*nvars + j;
                
                if (vI < 1e-10) {
                    phi_old[itr] = 1;
                }
                else {
                    phi_old[itr] = vJ / (vI + vJ);
                }
                
            }
            
        }
    
    }
    
} //end computePhiOld

/******************************************************************************/
template<typename ArgT>
void
FM::ES_solveCondLE(
    minitensor::Vector<ArgT, FM::MAX_TRNS>      const & A,
    minitensor::Vector<ArgT, FM::MAX_VRNT>            & X,
    minitensor::Vector<ArgT, FM::MAX_VRNT>      const & B,
    int                                        const nVars,
    int                                        const i,
    minitensor::Vector<ArgT, FM::MAX_TRNS>            & Aeye)
/******************************************************************************/
{
    Aeye = A;
    X = B;
    
    ArgT f;  //scale factor
    
    
    int ii;
    int ij;
    int ji;
    int jj;
    
    //The A matrix has a specific form of 
    
    //[ 11     ...    ...    ...    ii     ...    1N
    //  ...    22     ...    ...    2i     ...    2N
    //  ...    ...    33     ...    3i     ...    3N
    //  ...    ...    ...    jj     ji     ...    jN
    //  i1     i2     i3     ij     ii     iK     iN
    //  ...    ...    ...    ...    ki     kk     kN
    //  ...    ...    ...    ...    kN     ...    NN            
      
    
    
    //clear ith row   A(i,:) = A(i,:) - A(i,j)/A(j,j)*A(j,:)
    for (int j = 0; j < nVars; j++) {
        if (j!=i) {
            
            ii = i*nVars+i;
            ij = i*nVars+j;
            ji = j*nVars+i;
            jj = j*nVars+j;
            
            f = Aeye[ij]/Aeye[jj];
            
            Aeye[ij] += -Aeye[jj]*f;
            Aeye[ii] += -Aeye[ji]*f;
            
            X[i] += -X[j]*f;
        }
    }
    
    //clear ith column by clearing each row  A(j,:) = A(j,:) - A(j,i)/A(i,i)*A(i,:)
    for (int j = 0; j < nVars; j++) {
        if (j!=i) {
            
            ii = i*nVars+i;
            ij = i*nVars+j;
            ji = j*nVars+i;
            jj = j*nVars+j;
            
            f = Aeye[ji]/Aeye[ii];
            
            Aeye[ji] += -Aeye[ii]*f;
            
            X[j] += - X[i]*f;
        }
    }
    
    
    // set diagonals to 1 by unitizing each row. 
    for (int j = 0; j < nVars; j++) {
        ii = i*nVars+i;
        ij = i*nVars+j;
        ji = j*nVars+i;
        jj = j*nVars+j;
           
        f = 1/Aeye[jj];
            
        Aeye[jj] *= f;
            
        X[j] *= f;

    }
    


} //end solveCondLE






    
