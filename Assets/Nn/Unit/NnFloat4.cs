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
    using CalclationFloat4 = ICalculation<float4, NnFloat4.ForwardActivation, NnFloat4.BackError, NnFloat4.BackDelta>;

    public struct NnFloat4 : CalclationFloat4
    {

        //public ForwardActivation CreatActivation() => new ForwardActivation();
        //public BackError CreatError() => new BackError();
        //public BackDelta CreatDelta() => new BackDelta();

        public int NodesInUnit => unitLength;

        const int unitLength = 4;


        public struct ForwardActivation : CalclationFloat4.IForwardPropergationActivation
        {

            public float4 value { get; set; }


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


        public struct BackError : CalclationFloat4.IBackPropergationError<BackError>
        {

            public float4 value { get; set; }


            //// back last
            //public BackError CalculateError(float4 teach, float4 output)
            //{
            //    this = new BackError { value = -2.0f * (teach - output) };
            //}


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


        public struct BackDelta : CalclationFloat4.IBackPropergationDelta<BackDelta>
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
                NnWeights<float4> cxp_weithgs, NnWeights<float4> cxp_weithgs_delta, int i)
            {
                cxp_weithgs[i] -= cxp_weithgs_delta[i];
            }
        }




        //public struct ReLU : CalclationFloat4.IActivationFunction
        //{
        //    public float4 Activate(float4 u) => max(u, 0);
        //    public float4 Prime(float4 a) => sign(a);

        //    public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
        //    public void InitWeights(NnWeights<float4> weights) => weights.InitHe();
        //}
        //public struct Sigmoid : CalclationFloat4.IActivationFunction
        //{
        //    public float4 Activate(float4 u) => 1 / (1 + exp(-u));
        //    public float4 Prime(float4 a) => a * (1 - a);

        //    public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
        //    public void InitWeights(NnWeights<float4> weights) => weights.InitXivier();
        //}
        //public struct Affine : CalclationFloat4.IActivationFunction
        //{
        //    public float4 Activate(float4 u) => u;
        //    public float4 Prime(float4 a) => 1;

        //    public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
        //    public void InitWeights(NnWeights<float4> weights) => weights.InitRandom();
        //}


        public struct ReLU : CalclationFloat4.IActivationFunction
        {
            public float4 Activate(float4 u) => max(u, 0);
            public float4 Prime(float4 a) => sign(a);

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => new WeightInitializer().InitHe(weights);
        }
        public struct Sigmoid : CalclationFloat4.IActivationFunction
        {
            public float4 Activate(float4 u) => 1 / (1 + exp(-u));
            public float4 Prime(float4 a) => a * (1 - a);

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => new WeightInitializer().InitXivier(weights);
        }
        public struct Affine : CalclationFloat4.IActivationFunction
        {
            public float4 Activate(float4 u) => u;
            public float4 Prime(float4 a) => 1;

            public float4 CalculateError(float4 t, float4 o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float4> weights) => new WeightInitializer().InitRandom(weights);
        }



        public struct WeightInitializer// : IWeightInitialize
        {

            public unsafe void InitRandom(NnWeights<float4> ws)
            {
                var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());

                for (var i = 0; i < ws.lengthOfUnit; i++)
                {
                    ws[i] = rnd.NextFloat4();
                }

                setZeroAtLastPerWidth(ws);
            }

            public unsafe void InitXivier(NnWeights<float4> ws)
            {
                initX1X2(ws, 1.0f / sqrt((float4)ws.lengthOfUnit * 4));
            }

            public unsafe void InitHe(NnWeights<float4> ws)
            {
                initX1X2(ws, sqrt(2.0f / (float4)ws.lengthOfUnit * 4));
            }


            static unsafe void initX1X2(NnWeights<float4> ws, float4 std_deviation)
            {
                if (!ws.values.IsCreated) return;

                var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());
                var pi2 = 2.0f * math.PI;

                (float4 x1, float4 x2) calc_x1x2() => (
                    x1: sqrt(-2.0f * log(rnd.NextFloat4())) * std_deviation,
                    x2: pi2 * rnd.NextFloat4());


                for (var io = 0; io < ws.lengthOfUnit >> 1; io++)
                {
                    var (x1, x2) = calc_x1x2();

                    var v0 =
                    ws[io * 2 + 0] = x1 * cos(x2);
                    //if (isnan(v0)) Debug.Log($"{io * 2 + 0} {v0}");
                    //if (v0 == 0) Debug.Log($"{io * 2 + 0} {v0}");

                    var v1 =
                    ws[io * 2 + 1] = x1 * sin(x2);
                    //if (isnan(v1)) Debug.Log($"{io * 2 + 1} {v1}");
                    //if (v1 == 0) Debug.Log($"{io * 2 + 1} {v1}");
                }

                if ((ws.lengthOfUnit & 1) > 0)
                {
                    var (x1, x2) = calc_x1x2();

                    var v0 =
                    ws[ws.lengthOfUnit - 1] = x1 * sin(x2);
                    //if (isnan(v0)) Debug.Log($"{ws.length - 1} {v0}");
                    //if (v0 == 0) Debug.Log($"{ws.length - 1} {v0}");
                }

                setZeroAtLastPerWidth(ws);
            }

            static void setZeroAtLastPerWidth(NnWeights<float4> ws)
            {
                if (ws.widthLastRemain == 0) return;


                for (var i = 0; i < ws.lengthOfUnit; i += ws.widthOfUnit)
                {
                    setZeroLast_(i);
                }


                void setZeroLast_(int offset)
                {
                    var v = ws[offset + ws.widthOfUnit - 1];

                    ws[offset + ws.widthOfUnit - 1] = ws.widthLastRemain switch
                    {
                        1 => new float4(v.x, v.y, v.z, 0),
                        2 => new float4(v.x, v.y, 0, 0),
                        3 => new float4(v.x, 0, 0, 0),
                        _ => new float4(0, 0, 0, 0),
                    };
                }
            }
        }
    }
}
