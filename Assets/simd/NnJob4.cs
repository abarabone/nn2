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
        NnLayerForward4Job<ReLU> job1 = new();
        NnLayerForward4Job<Sigmoid> job2 = new();

        NnLayerBackLast4Job<ReLU> job5 = new();
        NnLayerBackLast4Job<Sigmoid> job6 = new();
        NnLayerBack4Job<ReLU> job3 = new();
        NnLayerBack4Job<Sigmoid> job4 = new();
    }





    [BurstCompile]
    public struct NnLayerForward4Job<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        [ReadOnly]
        public NnActivations prev_activations;

        [WriteOnly]
        public NnActivations curr_activations;

        [ReadOnly]
        public NnWeights cxp_weithgs;


        public void Execute(int ic4)
        {
            var sum = default(number);

            //var curr_index = ic * 4;
            var ip4 = 0;
            for (; ip4 < this.prev_activations.lengthOfUnits; ip4++)
            {
                var a = this.prev_activations[ip4];

                var ip = ip4 * 4;
                sum +=
                    a.xxxx * this.cxp_weithgs[ic4, ip + 0] +
                    a.yyyy * this.cxp_weithgs[ic4, ip + 1] +
                    a.zzzz * this.cxp_weithgs[ic4, ip + 2] +
                    a.wwww * this.cxp_weithgs[ic4, ip + 3];

                //sum += this.prev_activations[ip].mul(this.pxc_weithgs[ip, ic]);
            }
            var ip_ = ip4 * 4;
            sum += this.cxp_weithgs[ic4, ip_];

            this.curr_activations[ic4] = new TAct().Activate(sum);
        }
    }
    //static public class aaa
    //{
    //    static public number4
    //}

    // d  = -2(t - a) * a'
    // w -= d * a-
    [BurstCompile]
    public struct NnLayerBackLast4Job<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        // outputs
        [WriteOnly]
        public NativeArray<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights dst_cxp_weithgs;

        // for d calc
        [ReadOnly]
        public NativeArray<number> curr_trains;
        [ReadOnly]
        public NnActivations curr_activations;

        // for w calc
        [ReadOnly]//[DeallocateOnJobCompletion]
        public NnWeights cxp_weithgs;
        [ReadOnly]
        public NnActivations prev_activations;

        public float leaning_rate;


        public void Execute(int ic4)
        {
            var t = this.curr_trains[ic4];
            var o = this.curr_activations[ic4];

            var d = -2 * (t - o);
            d = d * new TAct().Prime(o);
            this.curr_ds[ic4] = d;


            var drate = d * this.leaning_rate;
            var ip4 = 0;
            for (; ip4 < this.prev_activations.lengthOfUnits; ip4++)
            {
                var a = this.prev_activations[ip4];

                var ip = ip4 * 4;
                this.dst_cxp_weithgs[ic4, ip + 0] = this.cxp_weithgs[ic4, ip + 0] - drate * a.xxxx;
                this.dst_cxp_weithgs[ic4, ip + 1] = this.cxp_weithgs[ic4, ip + 1] - drate * a.yyyy;
                this.dst_cxp_weithgs[ic4, ip + 2] = this.cxp_weithgs[ic4, ip + 2] - drate * a.zzzz;
                this.dst_cxp_weithgs[ic4, ip + 3] = this.cxp_weithgs[ic4, ip + 3] - drate * a.wwww;

                //this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
            }
            var ip_ = ip4 * 4;
            this.dst_cxp_weithgs[ic4, ip_] = this.cxp_weithgs[ic4, ip_] - drate;
        }
    }

    // d  = sum(d+ * w+) * a'
    // w -= d * a-
    [BurstCompile]
    public struct NnLayerBack4Job<TAct> : IJobParallelFor
        where TAct : struct, IActivationFunction
    {
        // outputs
        [WriteOnly]
        public NativeArray<number> curr_ds;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        public NnWeights dst_cxp_weithgs;


        // for d calc
        [ReadOnly]
        public NnWeights nxc_weithgs;
        [ReadOnly]
        public NativeArray<number> next_ds;
        [ReadOnly]
        public NnActivations curr_activations;

        // for w calc
        [ReadOnly]//[DeallocateOnJobCompletion]
        public NnWeights cxp_weithgs;
        [ReadOnly]
        public NnActivations prev_activations;

        public float leaning_rate;


        public void Execute(int ic4)
        {
            var sumd = default(number);
            for (var in4 = 0; in4 < this.next_ds.Length; in4++)
            {
                var ic = ic4 * 4;
                var nd = this.next_ds[in4];
                var dx = nd * this.nxc_weithgs[in4, ic + 0];
                var dy = nd * this.nxc_weithgs[in4, ic + 1];
                var dz = nd * this.nxc_weithgs[in4, ic + 2];
                var dw = nd * this.nxc_weithgs[in4, ic + 3];
                var md = new numbermt(dx, dy, dz, dw);
                var tmd = math.transpose(md);

                sumd += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
            }
            var d = sumd * new TAct().Prime(this.curr_activations[ic4]);
            this.curr_ds[ic4] = d;


            var drate = d * this.leaning_rate;
            var ip4 = 0;
            for (; ip4 < this.prev_activations.lengthOfUnits; ip4++)
            {
                var a = this.prev_activations[ip4];

                var ip = ip4 * 4;
                this.dst_cxp_weithgs[ic4, ip + 0] = this.cxp_weithgs[ic4, ip + 0] - drate * a.xxxx;
                this.dst_cxp_weithgs[ic4, ip + 1] = this.cxp_weithgs[ic4, ip + 1] - drate * a.yyyy;
                this.dst_cxp_weithgs[ic4, ip + 2] = this.cxp_weithgs[ic4, ip + 2] - drate * a.zzzz;
                this.dst_cxp_weithgs[ic4, ip + 3] = this.cxp_weithgs[ic4, ip + 3] - drate * a.wwww;

                //this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
            }
            var ip_ = ip4 * 4;
            this.dst_cxp_weithgs[ic4, ip_] = this.cxp_weithgs[ic4, ip_] - drate;
        }
    }
}