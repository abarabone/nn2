using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using static Unity.Mathematics.math;
using System.Runtime.ConstrainedExecution;
using Unity.VisualScripting;
using System.Diagnostics;

namespace nn
{
    using CalclationFloat = Calculation<float4, NnFloat4.ForwardActivation, NnFloat4.BackError, NnFloat4.BackDelta>;

    public class NnFloat4 : CalclationFloat
    {

        const int unitLength = 4;


        public struct ForwardActivation : CalclationFloat.IForwardPropergationActivation
        {

            public float4 value { get; set; }

            public int UnitLength => UnitLength;


            public void SumActivation(float4 a, NnWeights<float4> cxp_weithgs, int ic, int ip)
            {
                var ip_ = ip * unitLength;
                var ic_ = ic;

                this.value +=
                    a.xxxx * cxp_weithgs[ic_, ip_ + 0] +
                    a.yyyy * cxp_weithgs[ic_, ip_ + 1] +
                    a.zzzz * cxp_weithgs[ic_, ip_ + 2] +
                    a.wwww * cxp_weithgs[ic_, ip_ + 3];
            }

            public void SumBias(
                NnWeights<float4> cxp_weithgs, int ic, int ip)
            {
                var ip_ = ip * unitLength;

                this.value += cxp_weithgs[ic, ip_];
            }
        }


        public struct BackError : CalclationFloat.IBackPropergationError<BackError>
        {

            public float4 value { get; set; }


            // back last
            public BackError CalculateError(float4 teach, float4 output) =>
                this = new BackError { value = -2.0f * (teach - output) };


            // back other
            public void SumActivationError(
                float4 nextActivationDelta, NnWeights<float4> nxc_weithgs, int inext, int icurr)
            {
                var ic_ = icurr * unitLength;

                var nd = nextActivationDelta;
                var dx = nd * nxc_weithgs[inext, ic_ + 0];
                var dy = nd * nxc_weithgs[inext, ic_ + 1];
                var dz = nd * nxc_weithgs[inext, ic_ + 2];
                var dw = nd * nxc_weithgs[inext, ic_ + 3];

                var md = new Matrix4x4(dx, dy, dz, dw);
                var tmd = math.transpose(md);

                this.value += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
            }
        }


        public struct BackDelta : CalclationFloat.IBackPropergationDelta<BackDelta>
        {

            public float4 rated { get; set; }
            public float4 raw { get; set; }


            public BackDelta CalculateActivationDelta(
                float4 err, float4 o_prime, float learningRate)
            {
                var d = err * o_prime;
                var r = d * learningRate;

                return this = new BackDelta { rated = r, raw = d };
            }

            public void SetWeightActivationDelta(
                NnWeights<float4> dst_cxp_weithgs_delta, float4 a, int icurr, int iprev)
            {
                var ip_ = iprev * unitLength;
                var ic = icurr;

                var delta_rated = this.rated;
                dst_cxp_weithgs_delta[ic, ip_ + 0] = delta_rated * a.xxxx;
                dst_cxp_weithgs_delta[ic, ip_ + 1] = delta_rated * a.yyyy;
                dst_cxp_weithgs_delta[ic, ip_ + 2] = delta_rated * a.zzzz;
                dst_cxp_weithgs_delta[ic, ip_ + 3] = delta_rated * a.wwww;
            }
            public void SetWeightBiasDelta(
                NnWeights<float4> dst_cxp_weithgs_delta, int icurr, int iprev)
            {
                var ip_ = iprev * unitLength;
                var ic = icurr;

                var delta_rated = this.rated;
                dst_cxp_weithgs_delta[ic, ip_] = delta_rated;
            }

            public void ApplyDeltaToWeight(
                NnWeights<float4> cxp_weithgs, NnWeights<float4> cxp_weithgs_delta, int i) =>
                    cxp_weithgs[i] -= cxp_weithgs_delta[i];
        }




        public struct ReLU : CalclationFloat.IActivationFunction
        {
            public float4 Activate(float4 u) => max(u, 0);
            public float4 Prime(float4 a) => sign(a);

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => weights.InitHe();
        }
        public struct Sigmoid : CalclationFloat.IActivationFunction
        {
            public float4 Activate(float4 u) => 1 / (1 + exp(-u));
            public float4 Prime(float4 a) => a * (1 - a);

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => weights.InitXivier();
        }
        public struct Affine : CalclationFloat.IActivationFunction
        {
            public float4 Activate(float4 u) => u;
            public float4 Prime(float4 a) => 1;

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => weights.InitRandom();
        }

    }
}
