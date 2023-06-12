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
using System.Numerics;

namespace nn.simd
{

    using number = Unity.Mathematics.float4;
    using numbermt = Unity.Mathematics.float4x4;
    //using Unity.VisualScripting;




    class Nna4
    {
        NnLayerForwardJob<ReLU> job1 = new();
        NnLayerForwardJob<Sigmoid> job2 = new();

        NnLayerBackLastJob<ReLU> job5 = new();
        NnLayerBackLastJob<Sigmoid> job6 = new();
        NnLayerBackJob<ReLU> job3 = new();
        NnLayerBackJob<Sigmoid> job4 = new();
    }





    [BurstCompile]
    public struct NnLayerForwardJob<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        [ReadOnly]
        public NnActivations<number> prev_activations;

        [WriteOnly]
        public NnActivations<number> curr_activations;

        [ReadOnly]
        public NnWeights<number> cxp_weithgs;


        public unsafe void Execute(int ic_n)
        {
            var sum = default(number);

            var ip_n = 0;
            for (; ip_n < this.prev_activations.lengthOfUnits; ip_n++)
            {
                var a = this.prev_activations[ip_n];

                var ip = ip_n * (sizeof(number) >> 2);
                sum +=
                    a.xxxx * this.cxp_weithgs[ic_n, ip + 0] +
                    a.yyyy * this.cxp_weithgs[ic_n, ip + 1] +
                    a.zzzz * this.cxp_weithgs[ic_n, ip + 2] +
                    a.wwww * this.cxp_weithgs[ic_n, ip + 3];
            }
            var ip_ = ip_n * (sizeof(number) >> 2);
            sum += this.cxp_weithgs[ic_n, ip_];

            this.curr_activations[ic_n] = new TAct().Activate(sum);
        }
    }


    // d  = -2(t - a) * a'
    // w -= d * a-
    [BurstCompile]
    public struct NnLayerBackLastJob<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        // outputs
        [WriteOnly]
        public NnActivations<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights<number> dst_cxp_weithgs_delta;

        // for d calc
        [ReadOnly]
        public NativeArray<number> curr_trains;
        [ReadOnly]
        public NnActivations<number> curr_activations;

        // for w calc
        //[ReadOnly]//[DeallocateOnJobCompletion]
        //public NnWeights<number> cxp_weithgs;
        [ReadOnly]
        public NnActivations<number> prev_activations;

        public float leaning_rate;


        public unsafe void Execute(int ic_n)
        {
            var t = this.curr_trains[ic_n];
            var o = this.curr_activations[ic_n];

            var d = -2 * (t - o);
            d = d * new TAct().Prime(o);
            this.curr_ds[ic_n] = d;


            var drate = d * this.leaning_rate;
            var ip_n = 0;
            for (; ip_n < this.prev_activations.lengthOfUnits; ip_n++)
            {
                var a = this.prev_activations[ip_n];

                var ip = ip_n * (sizeof(number) >> 2);
                this.dst_cxp_weithgs_delta[ic_n, ip + 0] = drate * a.xxxx;
                this.dst_cxp_weithgs_delta[ic_n, ip + 1] = drate * a.yyyy;
                this.dst_cxp_weithgs_delta[ic_n, ip + 2] = drate * a.zzzz;
                this.dst_cxp_weithgs_delta[ic_n, ip + 3] = drate * a.wwww;
            }
            var ip_ = ip_n * (sizeof(number) >> 2);
            this.dst_cxp_weithgs_delta[ic_n, ip_] = drate;
        }
    }

    // d  = sum(d+ * w+) * a'
    // w -= d * a-
    [BurstCompile]
    public struct NnLayerBackJob<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        // outputs
        [WriteOnly]
        public NnActivations<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights<number> dst_cxp_weithgs_delta;


        // for d calc
        [ReadOnly]
        public NnWeights<number> nxc_weithgs;
        [ReadOnly]
        public NnActivations<number> next_ds;
        [ReadOnly]
        public NnActivations<number> curr_activations;

        // for w calc
        //[ReadOnly]//[DeallocateOnJobCompletion]
        //public NnWeights<number> cxp_weithgs;
        [ReadOnly]
        public NnActivations<number> prev_activations;

        public float leaning_rate;


        public unsafe void Execute(int ic_n)
        {
            var sumd = default(number);
            for (var in_n = 0; in_n < this.next_ds.lengthOfUnits; in_n++)
            {
                var ic = ic_n * (sizeof(number) >> 2);
                var nd = this.next_ds[in_n];
                var dx = nd * this.nxc_weithgs[in_n, ic + 0];
                var dy = nd * this.nxc_weithgs[in_n, ic + 1];
                var dz = nd * this.nxc_weithgs[in_n, ic + 2];
                var dw = nd * this.nxc_weithgs[in_n, ic + 3];
                var md = new numbermt(dx, dy, dz, dw);
                var tmd = math.transpose(md);

                sumd += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
            }
            var d = sumd * new TAct().Prime(this.curr_activations[ic_n]);
            this.curr_ds[ic_n] = d;


            var drate = d * this.leaning_rate;
            var ip_n = 0;
            for (; ip_n < this.prev_activations.lengthOfUnits; ip_n++)
            {
                var a = this.prev_activations[ip_n];

                var ip = ip_n * (sizeof(number) >> 2);
                this.dst_cxp_weithgs_delta[ic_n, ip + 0] = drate * a.xxxx;
                this.dst_cxp_weithgs_delta[ic_n, ip + 1] = drate * a.yyyy;
                this.dst_cxp_weithgs_delta[ic_n, ip + 2] = drate * a.zzzz;
                this.dst_cxp_weithgs_delta[ic_n, ip + 3] = drate * a.wwww;
            }
            var ip_ = ip_n * (sizeof(number) >> 2);
            this.dst_cxp_weithgs_delta[ic_n, ip_] = drate;
        }
    }



    [BurstCompile]
    public struct NnLayerUpdateWeightsJob : IJobParallelFor
    {

        //[WriteOnly]
        public NnWeights<number> cxp_weithgs;

        [ReadOnly]//[DeallocateOnJobCompletion]
        public NnWeights<number> cxp_weithgs_delta;


        public void Execute(int i_n)
        {
            this.cxp_weithgs[i_n] -= this.cxp_weithgs_delta[i_n];
        }
    }
}