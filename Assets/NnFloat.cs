//using System.Collections;
//using System.Collections.Generic;
//using System;
//using System.Linq;
//using UnityEngine;
//using Unity.Burst;
//using Unity.Collections;
//using Unity.Jobs;
//using Unity.Collections.LowLevel.Unsafe;
//using Unity.Mathematics;
//using static Unity.Mathematics.math;
//using System.Runtime.ConstrainedExecution;
//using Unity.VisualScripting;

//namespace nn
//{

//    public struct NnFloat : Nn<NnFloat, float>.ICalculatable
//    {

//        public int UnitLength => 1;



//        public struct ForwardActivation : Nn<NnFloat, float>.ICalculatable.IForwardPropergationActivation
//        {

//            public float value { get; set; }


//            public void SumActivation(float a, NnWeights<float> cxp_weithgs, int ic, int ip)
//            {
//                this.value +=
//                    a * cxp_weithgs[ic, ip + 0];
//            }

//            public void SumBias(
//                NnWeights<float> cxp_weithgs, int ic, int ip)
//            {
//                this.value += cxp_weithgs[ic, ip];
//            }
//        }


//        public struct BackErorr : Nn<NnFloat, float>.ICalculatable.IBackPropergationError<BackErorr>
//        {

//            public float value { get; set; }


//            // back last
//            public BackErorr CalculateError(float teach, float output) =>
//                this = new BackErorr { value = -2.0f * (teach - output) };


//            // back other
//            public void SumActivationError(
//                float nextActivationDelta, NnWeights<float> nxc_weithgs, int inext, int icurr)
//            {
//                var ic_ = icurr * new NnFloat().UnitLength;

//                var nd = nextActivationDelta;
//                var dx = nd * nxc_weithgs[inext, ic_ + 0];
//                var dy = nd * nxc_weithgs[inext, ic_ + 1];
//                var dz = nd * nxc_weithgs[inext, ic_ + 2];
//                var dw = nd * nxc_weithgs[inext, ic_ + 3];

//                var md = new Matrix4x4(dx, dy, dz, dw);
//                var tmd = math.transpose(md);

//                this.value += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
//            }
//        }


//        public struct BackDelta : Nn<NnFloat, float>.ICalculatable.IBackPropergationDelta<BackDelta>
//        {

//            public float rated { get; set; }
//            public float raw { get; set; }


//            public BackDelta CalculateActivationDelta(
//                float err, float o_prime, float learningRate)
//            {
//                var d = err * o_prime;
//                var r = d * learningRate;

//                return this = new BackDelta { rated = r, raw = d };
//            }

//            public void SetWeightActivationDelta(
//                NnWeights<float> dst_cxp_weithgs_delta, float a, int icurr, int iprev)
//            {
//                var ip_ = iprev * new NnFloat().UnitLength;
//                var ic = icurr;

//                var delta_rated = this.rated;
//                dst_cxp_weithgs_delta[ic, ip_ + 0] = delta_rated * a.xxxx;
//                dst_cxp_weithgs_delta[ic, ip_ + 1] = delta_rated * a.yyyy;
//                dst_cxp_weithgs_delta[ic, ip_ + 2] = delta_rated * a.zzzz;
//                dst_cxp_weithgs_delta[ic, ip_ + 3] = delta_rated * a.wwww;
//            }
//            public void SetWeightBiasDelta(
//                NnWeights<float> dst_cxp_weithgs_delta, int icurr, int iprev)
//            {
//                var ip_ = iprev * new NnFloat().UnitLength;
//                var ic = icurr;

//                var delta_rated = this.rated;
//                dst_cxp_weithgs_delta[ic, ip_] = delta_rated;
//            }

//            public void ApplyDeltaToWeight(
//                NnWeights<float> cxp_weithgs, NnWeights<float> cxp_weithgs_delta, int i) =>
//                    cxp_weithgs[i] -= cxp_weithgs_delta[i];
//        }


//    }

//}
