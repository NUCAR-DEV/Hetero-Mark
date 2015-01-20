/**
 * @struct CFlux2DCalculatorPars
 *
 * @brief A structure used to transfer calculation parameters
 */
typedef struct
{
    float time; // current time of the simulation
    float tau;  // current time step
} CFlux2DCalculatorPars;

/**
 * @struct CFluxVecPtr
 *
 * @brief A structure used to store flux pointers
 */
typedef struct
{
     __global float *pH;
     __global float *pU;
     __global float *pV;
} CFluxVecPtr;

/**
 * @struct CFluxVec
 *
 * @brief A structure used to store flux values
 */
typedef struct
{
    float H;
    float u;
    float v;
} CFluxVec;

/**
 * @struct CFlux2DLocalPtrs
 *
 * @brief A structure used to store flux pointers and other data
 */
typedef struct
{
    // Pointers to processed lines of data
    __global float*  pHbUp;                              // A pointer to upper line of bottom height
    __global float*  pHb;                                // A pointer to current line of bottom height
    __global float*  pHbDn;                              // A pointer to lower line of bottom height
    __global float*  pHbSqrtUp;                          // A pointer to upper line of bottom height square root
    __global float*  pHbSqrt;                            // A pointer to current line of bottom height square root
    __global float*  pHbSqrtDn;                          // A pointer to lower line of bottom height square root
    __global float*  pSpeedUp;                           // A pointer to uppper line of calculated wave speed
    __global float*  pSpeed;                             // A pointer to current line of calculated wave speed
    __global float*  pSpeedDn;                           // A pointer to lower line of calculated wave speed
    CFluxVecPtr inPtrUp;                                 // Pointers to the uppper line of H,U,V
    CFluxVecPtr inPtr;                                   // Pointers to the current line of H,U,V
    CFluxVecPtr inPtrDn;                                 // Pointers to the lower line of H,U,V

    // Output pointers
    CFluxVecPtr outPtr;                                   // Pointers to the current line of H,U,V for storing results

    // Flow parts
    CFluxVec flow;                                        // The flow for H,U,V

    // Current value
    CFluxVec cur;                                         // The current values for H,U,V
    float        curHb;                                   // The current value of the bottom height
    float        curHbSqrt;                               // The current value of the bottom height's square root
    float        curSpeed;                                // The current speed of the wave

} CFlux2DLocalPtrs;


/**
* @enum CRITICAL_SPEED
*
* @brief Defines negative and positive less and more critical speeds
*/
enum CRITICAL_SPEED
{
    SPEED_NEG_MORECRITICAL = 0,
    SPEED_NEG_LESSCRITICAL = 1,
    SPEED_POS_LESSCRITICAL = 2,
    SPEED_POS_MORECRITICAL = 3
};

/**
 * @struct CKernelData
 *
 * @brief A structure used to store global kernel parameters
 */
typedef struct
{
    float RcpGridStepW;
    float RcpGridStepH;
    CFlux2DCalculatorPars m_CalcParams;

    int SpeedCacheMatrixOffset;
    int SpeedCacheMatrixWidthStep;

    int HeightMapOffset;
    int HeightMapWidthStep;
    int UMapOffset;
    int UMapWidthStep;
    int VMapOffset;
    int VMapWidthStep;

    int HeightMapOffsetOut;
    int HeightMapWidthStepOut;
    int UMapOffsetOut;
    int UMapWidthStepOut;
    int VMapOffsetOut;
    int VMapWidthStepOut;

    int HeightMapBottomOffset;
    int HeightMapBottomWidthStep;

    int PrecompDataMapBottomOffset;
    int PrecompDataMapBottomWidthStep;

    float        Hmin;                          // The minimal allowed water bar height
    float        gravity;                       // A local copy of gravity value
    float        halfGravity;                   // The gravity multiplied by 0.5
    float        tau_div_deltaw;                // The time interval divided by grid cell width
    float        tau_div_deltah;                // The time interval divided by grid cell height
    float        tau_div_deltaw_quarter;        // A quarter of the time interval divided by the grid cell width
    float        tau_div_deltah_quarter;        // A quarter of the time interval divided by the grid cell height
    float        tau_mul_Cf;                    // The time interval multiplied by friction coefficient
    float        minNonZeroSpeed;               // The minimal value of water bar speed

} CKernelData;


///////////////////////////////////////////////////////////SCALAR CALCULATOR////////////////////////////////////////////////////////////////
/**
 * @fn CalcU_Neg_MoreCritical()
 *
 * @brief This method calculates the flow component related with a NEGATIVE water bar speed
 *        U component that has an absolute value that is MORE than the wave speed in the node
 */
void CalcU_Neg_MoreCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    /* ******************************************************************** *\
                                   Formulas used

                                    fH = dp[Hu]
                                    fU = dp[H*(0.5gH + u^2)]
                                    fV = dp[Huv]
                                    Ru = 0.5g(Hp+H)(hp-h)
    \* ******************************************************************** */

    // Now load new values for dp
    float H_  = pBuffers->inPtr.pH[1];
    float u_  = pBuffers->inPtr.pU[1];
    float v_  = pBuffers->inPtr.pV[1];
    float Hb_ = pBuffers->pHb[1];

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = -pBuffers->cur.H * pBuffers->cur.u;
    float tmpFlowU = -pBuffers->cur.H * (pData->halfGravity * pBuffers->cur.H + pBuffers->cur.u * pBuffers->cur.u);
    float tmpFlowV = -pBuffers->cur.H * pBuffers->cur.u * pBuffers->cur.v;

    // Calculate the positive portion of the water flow
    tmpFlowH += H_ * u_;
    tmpFlowU += H_ * (pData->halfGravity * H_ + u_ * u_);
    tmpFlowV += H_ * u_ * v_;
    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->halfGravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // For the final step, write the result to the flow output
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltaw;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltaw;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltaw;
}

/**
 * @fn CalcU_Pos_MoreCritical()
 *
 * @brief This method calculates the flow component related with a POSITIVE water bar speed U component
 *        that has an absolute value MORE than the wave speed in the node
 */
void CalcU_Pos_MoreCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    /* ******************************************************************** *\
                                   Formulas used

                                    fH = dm[Hu]
                                    fU = dm[H*(0.5gH + u^2)]
                                    fV = dm[Huv]
                                    Ru = 0.5g(H+Hm)(h-hm)
    \* ******************************************************************** */

    // Load values for the left node
    float H_  = pBuffers->inPtr.pH[-1];
    float u_  = pBuffers->inPtr.pU[-1];
    float v_  = pBuffers->inPtr.pV[-1];
    float Hb_ = pBuffers->pHb[-1];

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = pBuffers->cur.H * pBuffers->cur.u;
    float tmpFlowU = pBuffers->cur.H * (pData->halfGravity * pBuffers->cur.H + pBuffers->cur.u * pBuffers->cur.u);
    float tmpFlowV = pBuffers->cur.H * pBuffers->cur.u * pBuffers->cur.v;

    // Calculate the positive portion of the water flow
    tmpFlowH -= H_ * u_;
    tmpFlowU -= H_ * (pData->halfGravity * H_ + u_ * u_);
    tmpFlowV -= H_ * u_ * v_;

    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->halfGravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // Finally write to the output flows
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltaw;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltaw;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltaw;
}

/**
 * @fn CalcU_Neg_LessCritical()
 *
 * @brief This method calculates the flow component related with a NEGATIVE water bar speed U component
 *        that has an absolute value LESS than the wave speed in the node
 */
void CalcU_Neg_LessCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    /* ******************************************************************** *\
                                   Formulas used

                fH = 0.25 * ( dm[H(u+c)]        + dp[H(3u-c)] )
                fU = 0.25 * ( dm[H(u^2+2uc+gH)] + dp[H(3u^2-2uc+gH)]
                fV = 0.25 * ( dm[H(u+c)v]       + dp[H(3u-c)v] )
                Rh = 0.25 * ( -dm[h*sqrt(gh)]   + dp[h*sqrt(gh)] )
                Ru = 0.25 * ( (H+Hm)(h-hm)      + (Hp+H)(hp-h)  ) * g
    \* ******************************************************************** */

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = 2.0f * pBuffers->cur.H * (pBuffers->curSpeed - pBuffers->cur.u);
    float tmpFlowU = 2.0f * pBuffers->cur.H * (2.0f * pBuffers->curSpeed - pBuffers->cur.u) * pBuffers->cur.u;
    float tmpFlowV = tmpFlowH * pBuffers->cur.v;
    tmpFlowH -= 2.0f * pBuffers->curHbSqrt;

    // Load values for the left node
    float H_  = pBuffers->inPtr.pH[-1];
    float u_  = pBuffers->inPtr.pU[-1];
    float v_  = pBuffers->inPtr.pV[-1];
    float Hb_ = pBuffers->pHb[-1];
    float HbSqrt_ = pBuffers->pHbSqrt[-1];
    float speed_ = pBuffers->pSpeed[-1];

    // The part of the flow from the left node
    tmpFlowH -= H_ * (u_ + speed_);
    tmpFlowU -= H_ * (u_ * u_ + 2.0f * u_ * speed_ + pData->gravity * H_);
    tmpFlowV -= H_ * (u_ + speed_) * v_;
    tmpFlowH += HbSqrt_;
    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->gravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // Load new values for the right node
    H_  = pBuffers->inPtr.pH[1];
    u_  = pBuffers->inPtr.pU[1];
    v_  = pBuffers->inPtr.pV[1];
    Hb_ = pBuffers->pHb[1];
    HbSqrt_ = pBuffers->pHbSqrt[1];
    speed_ = pBuffers->pSpeed[1];

    // The part of the flow from the right node index
    tmpFlowH += H_ * (3.0f * u_ - speed_);
    tmpFlowU += H_ * (3.0f * u_ * u_ - 2.0f * u_ * speed_ + pData->gravity * H_);
    tmpFlowV += H_ * (3.0f * u_ - speed_) * v_;
    tmpFlowH += HbSqrt_;
    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->gravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // For the final step, write the result to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltaw_quarter;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltaw_quarter;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltaw_quarter;
}

/**
 * @fn CalcU_Pos_LessCritical()
 *
 * @brief This method calculates the flow component related with a POSITIVE water bar speed U component
 *        that has absolute value LESS than the wave speed in the node
 */
void CalcU_Pos_LessCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    /* ******************************************************************** *\
                                   Formulas used

                fH = 0.25 * ( dm[H(3u-c)]        + dp[H(u+c)] )
                fU = 0.25 * ( dm[H(3u^2+2uc+gH)] + dp[H(u^2-2uc+gH)]
                fV = 0.25 * ( dm[H(3u+c)v]       + dp[H(u-c)v] )
                Rh = 0.25 * ( -dm[h*sqrt(gh)]    + dp[h*sqrt(gh)] )
                Ru = 0.25 * ( (H+Hm)(h-hm)       + (Hp+H)(hp-h)  ) * g
    \* ******************************************************************** */

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = 2.0f * pBuffers->cur.H * (pBuffers->curSpeed + pBuffers->cur.u);
    float tmpFlowU = 2.0f * pBuffers->cur.H * (2.0f * pBuffers->curSpeed + pBuffers->cur.u) * pBuffers->cur.u;
    float tmpFlowV = tmpFlowH * pBuffers->cur.v;
    tmpFlowH -= 2.0f * pBuffers->curHbSqrt;

    // Now load new values for dm
    float H_  = pBuffers->inPtr.pH[-1];
    float u_  = pBuffers->inPtr.pU[-1];
    float v_  = pBuffers->inPtr.pV[-1];
    float Hb_ = pBuffers->pHb[-1];
    float HbSqrt_ = pBuffers->pHbSqrt[-1];
    float speed_ = pBuffers->pSpeed[-1];

    // The part of the flow from the negative index
    tmpFlowH -= H_ * (3.0f * u_ + speed_);
    tmpFlowU -= H_ * (3.0f * u_ * u_ + 2.0f * u_ * speed_ + pData->gravity * H_);
    tmpFlowV -= H_ * (3.0f * u_ + speed_) * v_;
    tmpFlowH += HbSqrt_;
    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->gravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // Now load new values for dp
    H_  = pBuffers->inPtr.pH[1];
    u_  = pBuffers->inPtr.pU[1];
    v_  = pBuffers->inPtr.pV[1];
    Hb_ = pBuffers->pHb[1];
    HbSqrt_ = pBuffers->pHbSqrt[1];
    speed_ = pBuffers->pSpeed[1];

    // The part of the flow from the negative index
    tmpFlowH += H_ * (u_ - speed_);
    tmpFlowU += H_ * (u_ * u_ - 2.0f * u_ * speed_ + pData->gravity * H_);
    tmpFlowV += H_ * (u_ - speed_) * v_;
    tmpFlowH += HbSqrt_;
    // In the final formula, Ru is taken with a minus
    tmpFlowU -= pData->gravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // For the final step, write the result to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltaw_quarter;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltaw_quarter;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltaw_quarter;
}

/**
 * @note V versions of the calculations are made from the corresponding U versions
 *       by looking into the formulas and replacing "U" with "V" and vise versa.
 *       U and V flows are also exchanged. +-1 offsets are replaced with stride.
 *       Rh, Ru, Rv (they are named Q values in all documents) remain unchanged
 */

/**
 * @fn CalcV_Neg_MoreCritical()
 *
 * @brief This method calculates the flow component related with a NEGATIVE water bar speed V component
 *        that has an absolute value MORE than the wave speed in the node
 */
void CalcV_Neg_MoreCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    // load values for the down node
    float H_  = pBuffers->inPtrDn.pH[0];
    float u_  = pBuffers->inPtrDn.pU[0];
    float v_  = pBuffers->inPtrDn.pV[0];
    float Hb_ = pBuffers->pHbDn[0];

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = -pBuffers->cur.H * pBuffers->cur.v;
    float tmpFlowU = -pBuffers->cur.H * pBuffers->cur.v * pBuffers->cur.u;
    float tmpFlowV = -pBuffers->cur.H * (pData->halfGravity * pBuffers->cur.H + pBuffers->cur.v * pBuffers->cur.v);

    // Calculate the positive portion of the water flow
    tmpFlowH += H_ * v_;
    tmpFlowU += H_ * v_ * u_;
    tmpFlowV += H_ * (pData->halfGravity * H_ + v_ * v_);

    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->halfGravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // For the final step, write the results to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltah;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltah;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltah;
}

/**
 * @fn CalcV_Pos_MoreCritical()
 *
 * @brief  This method calculates the flow component related with a POSITIVE water bar speed V component
 *         that has an absolute value MORE than the wave speed in the node
 */
void CalcV_Pos_MoreCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    // Load values for the up node
    float H_  = pBuffers->inPtrUp.pH[0];
    float u_  = pBuffers->inPtrUp.pU[0];
    float v_  = pBuffers->inPtrUp.pV[0];
    float Hb_ = pBuffers->pHbUp[0];

    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = pBuffers->cur.H * pBuffers->cur.v;
    float tmpFlowU = pBuffers->cur.H * pBuffers->cur.v * pBuffers->cur.u;
    float tmpFlowV = pBuffers->cur.H * (pData->halfGravity * pBuffers->cur.H + pBuffers->cur.v * pBuffers->cur.v);

    // Calculate the positive portion of the water flow
    tmpFlowH -= H_ * v_;
    tmpFlowU -= H_ * v_ * u_;
    tmpFlowV -= H_ * (pData->halfGravity * H_ + v_ * v_);

    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->halfGravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // For the final step, write the results to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltah;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltah;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltah;
}

/**
 * @fn CalcV_Neg_LessCritical()
 *
 * @brief  This method calculates the flow component related with a NEGATIVE water bar speed V component
 *         that has an absolute value LESS than the wave speed in the node
 */
void CalcV_Neg_LessCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = 2.0f * pBuffers->cur.H * (pBuffers->curSpeed - pBuffers->cur.v);
    float tmpFlowU = tmpFlowH * pBuffers->cur.u;
    float tmpFlowV = 2.0f * pBuffers->cur.H * (2.0f * pBuffers->curSpeed - pBuffers->cur.v) * pBuffers->cur.v;
    tmpFlowH -= 2.0f * pBuffers->curHbSqrt;

    // Now load new values for dm
    float H_  = pBuffers->inPtrUp.pH[0];
    float u_  = pBuffers->inPtrUp.pU[0];
    float v_  = pBuffers->inPtrUp.pV[0];
    float Hb_ = pBuffers->pHbUp[0];
    float speed_ = *pBuffers->pSpeedUp;

    // The part of the flow from a negative index
    tmpFlowH -= H_ * (v_ + speed_);
    tmpFlowU -= H_ * (v_ + speed_) * u_;
    tmpFlowV -= H_ * (v_ * v_ + 2.0f * v_ * speed_ + pData->gravity * H_);
    tmpFlowH += pBuffers->pHbSqrtUp[0];
    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->gravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // Now load new values for dp
    H_  = pBuffers->inPtrDn.pH[0];
    u_  = pBuffers->inPtrDn.pU[0];
    v_  = pBuffers->inPtrDn.pV[0];
    Hb_ = pBuffers->pHbDn[0];
    speed_ = *pBuffers->pSpeedDn;

    // The part of the flow from a negative index
    tmpFlowH += H_ * (3.0f * v_ - speed_);
    tmpFlowU += H_ * (3.0f * v_ - speed_) * u_;
    tmpFlowV += H_ * (3.0f * v_ * v_ - 2.0f * v_ * speed_ + pData->gravity * H_);
    tmpFlowH += pBuffers->pHbSqrtDn[0];

    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->gravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // Final step, write the results to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltah_quarter;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltah_quarter;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltah_quarter;
}

/**
 * @fn CalcV_Pos_LessCritical()
 *
 * @brief  This method calculates the flow component related with a POSITIVE water bar speed V component
 * that has an absolute value LESS than the wave speed in the node
 */
void CalcV_Pos_LessCritical(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    // First do the calculations with the center (H, u,v) point in the grid
    float tmpFlowH = 2.0f * pBuffers->cur.H * (pBuffers->curSpeed + pBuffers->cur.v);
    float tmpFlowU = tmpFlowH * pBuffers->cur.u;
    float tmpFlowV = 2.0f * pBuffers->cur.H * (2.0f * pBuffers->curSpeed + pBuffers->cur.v) * pBuffers->cur.v;
    tmpFlowH -= 2.0f * pBuffers->curHbSqrt;

    // Now load new values for dm
    float H_  = pBuffers->inPtrUp.pH[0];
    float u_  = pBuffers->inPtrUp.pU[0];
    float v_  = pBuffers->inPtrUp.pV[0];
    float Hb_ = pBuffers->pHbUp[0];

    float speed_ = *pBuffers->pSpeedUp;
    // The part of the flow from a negative index
    tmpFlowH -= H_ * (3.0f * v_ + speed_);
    tmpFlowU -= H_ * (3.0f * v_ + speed_) * u_;
    tmpFlowV -= H_ * (3.0f * v_ * v_ + 2.0f * v_ * speed_ + pData->gravity * H_);
    tmpFlowH += pBuffers->pHbSqrtUp[0];
    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->gravity * (pBuffers->cur.H + H_) * (pBuffers->curHb - Hb_);

    // Now load new values for dp
    H_  = pBuffers->inPtrDn.pH[0];
    u_  = pBuffers->inPtrDn.pU[0];
    v_  = pBuffers->inPtrDn.pV[0];
    Hb_ = pBuffers->pHbDn[0];
    speed_ = *pBuffers->pSpeedDn;

    // The part of the flow from a negative index
    tmpFlowH += H_ * (v_ - speed_);
    tmpFlowU += H_ * (v_ - speed_) * u_;
    tmpFlowV += H_ * (v_ * v_ - 2.0f * v_ * speed_ + pData->gravity * H_);
    tmpFlowH += pBuffers->pHbSqrtDn[0];
    // In the final formula, Ru is taken with a minus
    tmpFlowV -= pData->gravity * (H_ + pBuffers->cur.H) * (Hb_ - pBuffers->curHb);

    // Final step, write the results to the flow outputs
    pBuffers->flow.H += tmpFlowH * pData->tau_div_deltah_quarter;
    pBuffers->flow.u += tmpFlowU * pData->tau_div_deltah_quarter;
    pBuffers->flow.v += tmpFlowV * pData->tau_div_deltah_quarter;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
* @fn CorrectSpeed()
*
* @brief Used to correct the speed in order to avoid division by close to a zero value
*
* @param  speed - The corrected speed
*
* @param  pData - The structure with global kernel parameters
*
* @return The absolute value of the corrected speed value
*         Zero if the corrected speed is below the minimum
*/
float correctSpeed(float speed, __global CKernelData *pData)
{
    return (fabs(speed) >= pData->minNonZeroSpeed) ? speed : 0.0f;
}

/**
* @enum DRYSTATES
*
* @brief Used to specify node dry states
*/
enum DRYSTATES
{
    LT_MASK = (1 << 0),     // The left node is dry
    RT_MASK = (1 << 1),     // The right node is dry
    UP_MASK = (1 << 2),     // The top node is dry
    DN_MASK = (1 << 3),     // The bottom node is dry
    FULL_MASK = LT_MASK | RT_MASK | UP_MASK | DN_MASK
};
/**
* @fn Advance()
*
* @brief Used to shift all current pointers by given elements
*
* @param pLocalData - The structure with local data pointers
*
* @param step - The number elements to shift
*/
void Advance(CFlux2DLocalPtrs *pLocalData, int step)
{
    // Advance read pointers
    pLocalData->inPtr.pH += step;
    pLocalData->inPtr.pU += step;
    pLocalData->inPtr.pV += step;

    pLocalData->inPtrUp.pH += step;
    pLocalData->inPtrUp.pU += step;
    pLocalData->inPtrUp.pV += step;

    pLocalData->inPtrDn.pH += step;
    pLocalData->inPtrDn.pU += step;
    pLocalData->inPtrDn.pV += step;

    pLocalData->outPtr.pH += step;
    pLocalData->outPtr.pU += step;
    pLocalData->outPtr.pV += step;

    pLocalData->pHb+=step;
    pLocalData->pHbDn+=step;
    pLocalData->pHbUp+=step;
    pLocalData->pHbSqrt+=step;
    pLocalData->pHbSqrtUp+=step;
    pLocalData->pHbSqrtDn+=step;
    pLocalData->pSpeed+=step;
    pLocalData->pSpeedUp+=step;
    pLocalData->pSpeedDn+=step;
}


/**
* @fn ProcessNode()
*
* @brief Processes the current node. Calculates the flow for the node and changes the current speed and height of the water bar
*/
void ProcessNode(CFlux2DLocalPtrs *pBuffers, __global CKernelData *pData)
{
    // calculate each pixel
    int state=0;

    // Load current values
    pBuffers->cur.H = *pBuffers->inPtr.pH;
    pBuffers->cur.u     = *pBuffers->inPtr.pU;
    pBuffers->cur.v     = *pBuffers->inPtr.pV;
    pBuffers->curHb     = *pBuffers->pHb;
    pBuffers->curHbSqrt = *pBuffers->pHbSqrt;

    // Calculate critical speed
    pBuffers->curSpeed = *pBuffers->pSpeed;

    // Init flows
    pBuffers->flow.H  = 0.0f;
    pBuffers->flow.u  = 0.0f;
    pBuffers->flow.v  = 0.0f;

    // Compute the state
    if (pBuffers->cur.H == 0.0f)
    {
        if (pBuffers->inPtr.pH[-1] == 0.0f)
        {
            state |= LT_MASK;
        }
        if (pBuffers->inPtr.pH[1] == 0.0f)
        {
            state |= RT_MASK;
        }
        if (pBuffers->inPtrUp.pH[0] == 0.0f)
        {
            state |= UP_MASK;
        }
        if (pBuffers->inPtrDn.pH[0] == 0.0f)
        {
            state |= DN_MASK;
        }
        if (state == FULL_MASK)
        {
            *pBuffers->outPtr.pH = 0.0f;
            *pBuffers->outPtr.pU = 0.0f;
            *pBuffers->outPtr.pV = 0.0f;
            return;
        }
    }


    if (state == 0)
    {
        // Calculate the U component of flows
        if (pBuffers->inPtr.pU[0] >= 0.0f)
        {
            if (pBuffers->inPtr.pU[0] > pBuffers->curSpeed)
            {
                CalcU_Pos_MoreCritical(pBuffers, pData);
            }
            else
            {
                CalcU_Pos_LessCritical(pBuffers, pData);
            }
        }
        else
        {
            if (pBuffers->inPtr.pU[0] < -pBuffers->curSpeed)
            {
                CalcU_Neg_MoreCritical(pBuffers, pData);
            }
            else
            {
                CalcU_Neg_LessCritical(pBuffers, pData);
            }
        }

        // Calculate the V component of flows
        if (pBuffers->inPtr.pV[0] >= 0.0f)
        {
            if (pBuffers->inPtr.pV[0] > pBuffers->curSpeed)
            {
                CalcV_Pos_MoreCritical(pBuffers, pData);
            }
            else
            {
                CalcV_Pos_LessCritical(pBuffers, pData);
            }
        }
        else
        {
            if (pBuffers->inPtr.pV[0] < -pBuffers->curSpeed)
            {
                CalcV_Neg_MoreCritical(pBuffers, pData);
            }
            else
            {
                CalcV_Neg_LessCritical(pBuffers, pData);
            }
        }
    }
    else // State is NOT zero
    {
        // First, check right and left
        switch (state & (LT_MASK | RT_MASK))
        {
        case 0: // Both neighbour node are active.
            {
                CalcU_Pos_LessCritical(pBuffers, pData);
                break;
            }
        case LT_MASK: // The left node is dry. Flow goes from the right only.
            {
                CalcU_Neg_MoreCritical(pBuffers, pData);
                break;
            }
        case RT_MASK:// The right node is dry. Flow goes from left only.
            {
                CalcU_Pos_MoreCritical(pBuffers, pData);
                break;
            }
        }

        // Then check up and down
        switch (state & (UP_MASK | DN_MASK))
        {
        case 0: // Both neighbour node are active.
            {
                CalcV_Pos_LessCritical(pBuffers, pData);
                break;
            }
        case UP_MASK: // The up node is dry. Flow goes from down only.
            {
                CalcV_Neg_MoreCritical(pBuffers, pData);
                break;
            }
        case DN_MASK: // The right node is dry. Flow goes from left only.
            {
                CalcV_Pos_MoreCritical(pBuffers, pData);
                break;
            }
        }
    }

    // NOTE: flow values are already multiplied by corresponding values

    // Calculate new H value
    float Hnew = pBuffers->inPtr.pH[0] - pBuffers->flow.H;
    if (Hnew >= 0.0f)
    {
        *pBuffers->outPtr.pH = Hnew;
        // Calculate new (u,v) values
        float sq = pData->tau_mul_Cf * sqrt(pBuffers->inPtr.pU[0] * pBuffers->inPtr.pU[0] + pBuffers->inPtr.pV[0] * pBuffers->inPtr.pV[0]);
        float rcpHnew = 1.0f / max(pData->Hmin, Hnew);
        *pBuffers->outPtr.pU = correctSpeed( rcpHnew * ((pBuffers->cur.H - sq) * pBuffers->cur.u - pBuffers->flow.u),  pData);
        *pBuffers->outPtr.pV = correctSpeed( rcpHnew * ((pBuffers->cur.H - sq) * pBuffers->cur.v - pBuffers->flow.v) , pData);
    }
    else
    {
        *pBuffers->outPtr.pH = 0.0f;
        *pBuffers->outPtr.pU = 0.0f;
        *pBuffers->outPtr.pV = 0.0f;
    }
}// End of ProcessNode

/**
* @fn CalcJobRange()
*
* @brief This method is used to calculate job range for each tile of a grid
*
*/
void CalcJobRange(int task, int taskNum, int jobNum, int* pJobStart, int* pJobEnd)
{
    int step = jobNum/taskNum;
    int rest = jobNum-taskNum*step;
    int j0,j1;

    if(task<rest)
    {
        /* big job basket */
        j0 = task*(step+1);
        j1 = j0+step;
    }
    else
    {
        /* small job basket */
        j0 = task*step+rest;
        j1 = j0+step-1;
    }
    if (j0<jobNum && j1<jobNum)
    {
        pJobStart[0] = j0;
        pJobEnd[0] = j1;
    }
    else
    {
        pJobStart[0] = -1;
        pJobEnd[0] = -2;
    }
}/* CalcJobRange */


/**
* @fn ProcessTile()
*
* @brief This method is used to process a tile of a grid
*
*/
__kernel void ProcessTile( __global char*  m_SpeedCacheMatrix,
                           __global char*  m_heightMap,
                           __global char*  m_uMap,
                           __global char*  m_vMap,
                           __global char*  m_heightMapOut,
                           __global char*  m_uMapOut,
                           __global char*  m_vMapOut,
                           __global char*  m_heightMapBottom,
                           __global char*  m_precompDataMapBottom,
                           __global CKernelData*  pData,
                          int width, int height)
{
    int       i;
    __global float*    pH = 0;
    int x0 = 0;
    int dimW = width;


    int y0;
    int y1;
    int dimH;
    CalcJobRange(get_global_id(0), get_global_size(0), height, &y0, &y1);
    dimH = y1-y0+1;

    CFlux2DLocalPtrs LocalData;

    pData->tau_div_deltaw  = pData->m_CalcParams.tau * pData->RcpGridStepW;
    pData->tau_div_deltah  = pData->m_CalcParams.tau * pData->RcpGridStepH;

    pData->tau_div_deltaw_quarter  = pData->tau_div_deltaw * 0.25f;
    pData->tau_div_deltah_quarter  = pData->tau_div_deltah * 0.25f;


    LocalData.pSpeedUp = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + (0+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);///m_SpeedCacheMatrix.GetLinePtr(0);
    LocalData.pSpeed = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + (1+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);///m_SpeedCacheMatrix.GetLinePtr(1);
    LocalData.pSpeedDn = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + (2+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);///m_SpeedCacheMatrix.GetLinePtr(2);

    // Calculate -1 line of speed that will be used in future
    pH = (__global float*)(m_heightMap + pData->HeightMapOffset + (y0-1)*pData->HeightMapWidthStep) + x0; //pH = pG->GetCurrentSurface().m_heightMap.GetLinePtr(y0-1)+x0;
    for(i=-1;i<dimW+1;++i)
    {
        LocalData.pSpeedUp[i] = sqrt(pData->gravity*pH[i]);
    }

    // Calculate 0 line of speed that will be used in future
    pH = (__global float*)(m_heightMap + pData->HeightMapOffset + (y0)*pData->HeightMapWidthStep) + x0; //pH = pG->GetCurrentSurface().m_heightMap.GetLinePtr(y0)+x0;
    for(i=-1;i<dimW+1;++i)
    {
        LocalData.pSpeed[i] = sqrt(pData->gravity*pH[i]);
    }

    // Start loop, for every line in the tile
    for (int y = 0; y < dimH; y++)
    {
        {
            LocalData.inPtr.pH = ( __global float*)(m_heightMap + pData->HeightMapOffset + (y0+y)*pData->HeightMapWidthStep) + x0;
            LocalData.inPtr.pU = ( __global float*)(m_uMap + pData->UMapOffset + (y0+y)*pData->UMapWidthStep) + x0;
            LocalData.inPtr.pV = ( __global float*)(m_vMap + pData->VMapOffset + (y0+y)*pData->VMapWidthStep) + x0;

            LocalData.inPtrUp.pH = ( __global float*)(m_heightMap + pData->HeightMapOffset + (y0+y-1)*pData->HeightMapWidthStep) + x0;
            LocalData.inPtrUp.pU = ( __global float*)(m_uMap + pData->UMapOffset + (y0+y-1)*pData->UMapWidthStep) + x0;
            LocalData.inPtrUp.pV = ( __global float*)(m_vMap + pData->VMapOffset + (y0+y-1)*pData->VMapWidthStep) + x0;

            LocalData.inPtrDn.pH = ( __global float*)(m_heightMap + pData->HeightMapOffset + (y0+y+1)*pData->HeightMapWidthStep) + x0;
            LocalData.inPtrDn.pU = ( __global float*)(m_uMap + pData->UMapOffset + (y0+y+1)*pData->UMapWidthStep) + x0;
            LocalData.inPtrDn.pV = ( __global float*)(m_vMap + pData->VMapOffset + (y0+y+1)*pData->VMapWidthStep) + x0;

            LocalData.outPtr.pH = ( __global float*)(m_heightMapOut + pData->HeightMapOffsetOut + (y0+y)*pData->HeightMapWidthStepOut) + x0;
            LocalData.outPtr.pU = ( __global float*)(m_uMapOut + pData->UMapOffsetOut + (y0+y)*pData->UMapWidthStepOut) + x0;
            LocalData.outPtr.pV = ( __global float*)(m_vMapOut + pData->VMapOffsetOut + (y0+y)*pData->VMapWidthStepOut) + x0;

            LocalData.pHbUp = ( __global float*)(m_heightMapBottom + pData->HeightMapBottomOffset + (y0+y-1)*pData->HeightMapBottomWidthStep) + x0;

            LocalData.pHb = ( __global float*)(m_heightMapBottom + pData->HeightMapBottomOffset + (y0+y)*pData->HeightMapBottomWidthStep) + x0;

            LocalData.pHbDn = ( __global float*)(m_heightMapBottom + pData->HeightMapBottomOffset + (y0+y+1)*pData->HeightMapBottomWidthStep) + x0;

            LocalData.pHbSqrtUp = ( __global float*)(m_precompDataMapBottom + pData->PrecompDataMapBottomOffset + (y0+y-1)*pData->PrecompDataMapBottomWidthStep) + x0;

            LocalData.pHbSqrt = ( __global float*)(m_precompDataMapBottom + pData->PrecompDataMapBottomOffset + (y0+y)*pData->PrecompDataMapBottomWidthStep) + x0;

            LocalData.pHbSqrtDn = ( __global float*)(m_precompDataMapBottom + pData->PrecompDataMapBottomOffset + (y0+y+1)*pData->PrecompDataMapBottomWidthStep) + x0;

            LocalData.pSpeedUp = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + (y%3+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);

            LocalData.pSpeed = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + ((y+1)%3+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);

            LocalData.pSpeedDn = ( __global float*)(m_SpeedCacheMatrix + pData->SpeedCacheMatrixOffset + ((y+2)%3+get_global_id(0)*3)*pData->SpeedCacheMatrixWidthStep);
        }



        // Calculate +1 line of speed that will be used in future
        for(i=-1;i<dimW+1;++i)
        {
            LocalData.pSpeedDn[i] = sqrt(pData->gravity*LocalData.inPtrDn.pH[i]);
        }

        // For every pixel in the line
        for (int x = 0; x < dimW; x++)
        {
            // calculate each pixel
            ProcessNode(&LocalData, pData);
            // And then advance the pointers
            Advance(&LocalData,1);
        }// End of next pixel in the line
    }// End of next line of the tile
}// End of ProcessTile

