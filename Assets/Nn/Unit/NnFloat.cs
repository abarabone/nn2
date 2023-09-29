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
//using System.Diagnostics;
using static nn.NnFloat;
using static Unity.Mathematics.Random;

namespace nn
{
    using CalclationFloat = ICalculation<float, NnFloat.ForwardActivation, NnFloat.BackError, NnFloat.BackDelta>;

    public struct NnFloat : CalclationFloat
    {

        //public ForwardActivation CreatActivation() => new ForwardActivation();
        //public BackError CreatError() => new BackError();
        //public BackDelta CreatDelta() => new BackDelta();

        public int NodesInUnit => unitLength;

        const int unitLength = 1;


        public struct ForwardActivation : CalclationFloat.IForwardPropergationActivation
        {

            public float value { get; set; }


            public void SumActivation(float a, NnWeights<float> cxp_weights, int ic, int ip)
            {
                this.value += a * cxp_weights[ic, ip];
            }

            public void SumBias(NnWeights<float> cxp_weights, int ic, int ip)
            {
                this.value += cxp_weights[ic, ip];
            }
        }


        public struct BackError : CalclationFloat.IBackPropergationError<BackError>
        {

            public float value { get; set; }


            //// last only
            //public BackError CalculateError(float teach, float output)
            //{
            //    return this = new BackError { value = -2.0f * (teach - output) };
            //}

            // not last
            public void SumActivationError(
                float nextActivationDelta, NnWeights<float> nxc_weights, int inext, int icurr)
            {
                var nd = nextActivationDelta;
                var dx = nd * nxc_weights[inext, icurr];

                this.value += dx;
            }
        }


        public struct BackDelta : CalclationFloat.IBackPropergationDelta<BackDelta>
        {

            public float rated { get; set; }
            public float raw { get; set; }


            public BackDelta CalculateActivationDelta(
                float err, float o_prime, float learningRate)
            {
                var d = err * o_prime;
                var r = d * learningRate;

                return this = new BackDelta { rated = r, raw = d };
            }

            public void SetWeightActivationDelta(
                NnWeights<float> dst_cxp_weights_delta, float a, int icurr, int iprev)
            {
                var delta_rated = this.rated;
                dst_cxp_weights_delta[icurr, iprev] = delta_rated * a;
            }
            public void SetWeightBiasDelta(
                NnWeights<float> dst_cxp_weights_delta, int icurr, int iprev)
            {
                var delta_rated = this.rated;
                dst_cxp_weights_delta[icurr, iprev] = delta_rated;
            }

            public void ApplyDeltaToWeight(
                NnWeights<float> cxp_weights, NnWeights<float> cxp_weights_delta, int i)
            {
                cxp_weights[i] -= cxp_weights_delta[i];
            }
        }




        public struct ReLU : CalclationFloat.IActivationFunction
        {
            public float Activate(float u) => max(u, 0);
            public float Prime(float a) => sign(a);

            public float CalculateError(float t, float o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float> weights) => new WeightInitializer().InitHe(weights);
        }
        public struct Sigmoid : CalclationFloat.IActivationFunction
        {
            public float Activate(float u) => 1 / (1 + exp(-u));
            public float Prime(float a) => a * (1 - a);

            public float CalculateError(float t, float o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float> weights) => new WeightInitializer().InitXivier(weights);
        }
        public struct Affine : CalclationFloat.IActivationFunction
        {
            public float Activate(float u) => u;
            public float Prime(float a) => 1;

            public float CalculateError(float t, float o) => -2.0f * (t - o);
            public void InitWeights(NnWeights<float> weights) => new WeightInitializer().InitRandom(weights);
        }



        public struct WeightInitializer// : IWeightInitialize
        {

            public unsafe void InitRandom(NnWeights<float> ws)
            {
                var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());

                for (var i = 0; i < ws.values.Length; i++)
                {
                    ws[i] = rnd.NextFloat();
                }
            }

            public unsafe void InitXivier(NnWeights<float> ws)
            {
                initX1X2(ws, 1.0f / sqrt((float)ws.values.Length));
            }

            public unsafe void InitHe(NnWeights<float> ws)
            {
                initX1X2(ws, sqrt(2.0f / (float)ws.values.Length));
            }


            static unsafe void initX1X2(NnWeights<float> ws, float std_deviation)
            {
                if (!ws.values.IsCreated) return;
                
                var rnd = new Unity.Mathematics.Random((uint)ws.values.GetUnsafePtr());
                var pi2 = 2.0f * math.PI;

                (float x1, float x2) calc_x1x2() => (
                    x1: sqrt(-2.0f * log(rnd.NextFloat())) * std_deviation,
                    x2: pi2 * rnd.NextFloat());


                for (var io = 0; io < ws.lengthOfUnit >> 1; io++)
                {
                    var (x1, x2) = calc_x1x2();

                    var v0 =
                    ws[io * 2 + 0] = x1 * cos(x2);
                    if (isnan(v0)) Debug.Log($"{io * 2 + 0} {v0}");
                    if (v0 == 0) Debug.Log($"{io * 2 + 0} {v0}");

                    var v1 =
                    ws[io * 2 + 1] = x1 * sin(x2);
                    if (isnan(v1)) Debug.Log($"{io * 2 + 1} {v1}");
                    if (v1 == 0) Debug.Log($"{io * 2 + 1} {v1}");
                }

                if ((ws.lengthOfUnit & 1) > 0)
                {
                    var (x1, x2) = calc_x1x2();

                    var v0 =
                    ws[ws.lengthOfUnit - 1] = x1 * sin(x2);
                    if (isnan(v0)) Debug.Log($"{ws.lengthOfUnit - 1} {v0}");
                    if (v0 == 0) Debug.Log($"{ws.lengthOfUnit - 1} {v0}");
                }
            }
        }
    }
}
