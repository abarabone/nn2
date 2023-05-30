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

using number = System.Single;
using number4 = Unity.Mathematics.float4;
using number4x4 = Unity.Mathematics.float4x4;


//static class Nna
//{
//    static NnLayerJob<ReLU> job1 = new ();
//    static NnLayerJob<Sigmoid> job2 = new ();

//    static NnLayerBackLastJob<ReLU> job5 = new();
//    static NnLayerBackLastJob<Sigmoid> job6 = new();
//    static NnLayerBackJob<ReLU> job3 = new();
//    static NnLayerBackJob<Sigmoid> job4 = new();
//}
class Nna
{
    NnLayerJob<ReLU> job1 = new();
    NnLayerJob<Sigmoid> job2 = new();

    NnLayerBackLastJob<ReLU> job5 = new();
    NnLayerBackLastJob<Sigmoid> job6 = new();
    NnLayerBackJob<ReLU> job3 = new();
    NnLayerBackJob<Sigmoid> job4 = new();
}

[BurstCompile]
public struct NnLayerJob<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction
{
    [ReadOnly]
    public NnActivations<number> prev_activations;

    [WriteOnly]
    public NnActivations<number> curr_activations;

    [ReadOnly]
    public NnWeights<number> pxc_weithgs;


    public void Execute(int curr_index)
    {
        var sum = default(number);

        var ic = curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            sum += this.prev_activations[ip] * this.pxc_weithgs[ip, ic];
            //Debug.Log($"fwd p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} * a-:{this.prev_activations[ip]}");
        }

        this.curr_activations[curr_index] = new TAct().Activate(sum);
    }
}


//d2 = (t - a2)
//w2 -= d2 * a1

//d1 = d2 * a1'
//w1 -= d1 * a0


// d  = -2(t - a) * a'
// w -= d * a-
[BurstCompile]
public struct NnLayerBackLastJob<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction
{
    // outputs
    [WriteOnly]
    public NativeArray<number> curr_ds;
    [WriteOnly][NativeDisableParallelForRestriction]
    public NnWeights<number> dst_pxc_weithgs;

    // for d calc
    [ReadOnly]
    public NativeArray<number> curr_trains;
    [ReadOnly]
    public NnActivations<number> curr_activations;

    // for w calc
    [ReadOnly]//[DeallocateOnJobCompletion]
    public NnWeights<number> pxc_weithgs;
    [ReadOnly]
    public NnActivations<number> prev_activations;

    public float leaning_rate;


    public void Execute(int curr_index)
    {
        var t = this.curr_trains[curr_index];
        var a = this.curr_activations[curr_index];

        var d = -2 * (t - a);
        //Debug.Log($"back last d sum c{curr_index} x n{0} -> t:{t} a:{a} dl:{d}");
        d = d * new TAct().Prime(a);
        var drate = d * this.leaning_rate;


        var ic = curr_index;
        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
            //Debug.Log($"back last sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
        }

        this.curr_ds[curr_index] = d;
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
    public NativeArray<number> curr_ds;
    [WriteOnly][NativeDisableParallelForRestriction]
    public NnWeights<number> dst_pxc_weithgs;


    // for d calc
    [ReadOnly]
    public NnWeights<number> cxn_weithgs;
    [ReadOnly]
    public NativeArray<number> next_ds;
    [ReadOnly]
    public NnActivations<number> curr_activations;

    // for w calc
    [ReadOnly]//[DeallocateOnJobCompletion]
    public NnWeights<number> pxc_weithgs;
    [ReadOnly]
    public NnActivations<number> prev_activations;

    public float leaning_rate;


    public void Execute(int curr_index)
    {
        var ic = curr_index;

        var sumd = default(number);
        for (var inext = 0; inext < this.next_ds.Length; inext++)
        {
            sumd += this.next_ds[inext] * this.cxn_weithgs[ic, inext];
            //Debug.Log($"back d sum c{ic} x n{inext} -> w:{this.cxn_weithgs[ic, inext]} * d+:{this.next_ds[inext]}");
        }
        var d = sumd * new TAct().Prime(this.curr_activations[curr_index]);
        var drate = d * this.leaning_rate;


        for (var ip = 0; ip < this.prev_activations.lengthWithBias; ip++)
        {
            this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
            //Debug.Log($"back sub weight p{ip} x c{ic} -> w:{this.pxc_weithgs[ip, ic]} a-:{this.prev_activations[ip]}");
        }

        this.curr_ds[curr_index] = d;
    }
}







[BurstCompile]
public struct NnLayerForward4Job<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction4
{
    [ReadOnly]
    public NnActivations4 prev_activations;

    [WriteOnly]
    public NnActivations4 curr_activations;

    [ReadOnly]
    public NnWeights4 pxc_weithgs;


    public void Execute(int ic4)
    {
        var sum = default(number4);

        //var curr_index = ic * 4;
        for (var ip4 = 0; ip4 < this.prev_activations.lengthOfUnits; ip4++)
        {
            var a = this.prev_activations[ip4];

            var ip = ip4 * 4;
            sum +=
                a.xxxx * this.pxc_weithgs[ip + 0, ic4] +
                a.yyyy * this.pxc_weithgs[ip + 1, ic4] +
                a.zzzz * this.pxc_weithgs[ip + 2, ic4] +
                a.wwww * this.pxc_weithgs[ip + 3, ic4];

            //sum += this.prev_activations[ip].mul(this.pxc_weithgs[ip, ic]);
        }
        sum += this.pxc_weithgs[this.prev_activations.lengthOfUnits * 4, ic4];

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
    where TAct : struct, IActivationFunction4
{
    // outputs
    [WriteOnly]
    public NativeArray<number4> curr_ds;
    [WriteOnly]
    [NativeDisableParallelForRestriction]
    public NnWeights4 dst_pxc_weithgs;

    // for d calc
    [ReadOnly]
    public NativeArray<number4> curr_trains;
    [ReadOnly]
    public NnActivations<number4> curr_activations;

    // for w calc
    [ReadOnly]//[DeallocateOnJobCompletion]
    public NnWeights4 pxc_weithgs;
    [ReadOnly]
    public NnActivations4 prev_activations;

    public float leaning_rate;


    public void Execute(int ic4)
    {
        var t = this.curr_trains[ic4];
        var o = this.curr_activations[ic4];

        var d = -2 * (t - o);
        d = d * new TAct().Prime(o);

        var drate = d * this.leaning_rate;
        for (var ip4 = 0; ip4 < this.prev_activations.lengthOfUnits; ip4++)
        {
            var a = this.prev_activations[ip4];

            var ip = ip4 * 4;
            this.dst_pxc_weithgs[ip + 0, ic4] = this.pxc_weithgs[ip + 0, ic4] - drate * a.xxxx;
            this.dst_pxc_weithgs[ip + 1, ic4] = this.pxc_weithgs[ip + 1, ic4] - drate * a.yyyy;
            this.dst_pxc_weithgs[ip + 2, ic4] = this.pxc_weithgs[ip + 2, ic4] - drate * a.zzzz;
            this.dst_pxc_weithgs[ip + 3, ic4] = this.pxc_weithgs[ip + 3, ic4] - drate * a.wwww;

            //this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
        }
        //this.dst_pxc_weithgs[, ic4] = this.pxc_weithgs[, ic4] - drate;

        this.curr_ds[ic4] = d;
    }
}

// d  = sum(d+ * w+) * a'
// w -= d * a-
[BurstCompile]
public struct NnLayerBack4Job<TAct> : IJobParallelFor
    where TAct : struct, IActivationFunction4
{
    // outputs
    [WriteOnly]
    public NativeArray<number4> curr_ds;
    [WriteOnly]
    [NativeDisableParallelForRestriction]
    public NnWeights4 dst_pxc_weithgs;


    // for d calc
    [ReadOnly]
    public NnWeights4 cxn_weithgs;
    [ReadOnly]
    public NativeArray<number4> next_ds;
    [ReadOnly]
    public NnActivations4 curr_activations;

    // for w calc
    [ReadOnly]//[DeallocateOnJobCompletion]
    public NnWeights4 pxc_weithgs;
    [ReadOnly]
    public NnActivations4 prev_activations;

    public float leaning_rate;


    public void Execute(int ic4)
    {
        var sumd = default(number4);
        for (var in4 = 0; in4 < this.next_ds.Length; in4++)
        {
            var ic = ic4 * 4;
            var nd = this.next_ds[in4];
            var dx = nd * this.cxn_weithgs[ic + 0, in4];
            var dy = nd * this.cxn_weithgs[ic + 1, in4];
            var dz = nd * this.cxn_weithgs[ic + 2, in4];
            var dw = nd * this.cxn_weithgs[ic + 3, in4];
            var md = new number4x4(dx, dy, dz, dw);
            var tmd = math.transpose(md);

            sumd += tmd.c0 + tmd.c1 + tmd.c2 + tmd.c3;
        }
        var d = sumd * new TAct().Prime(this.curr_activations[ic4]);


        var drate = d * this.leaning_rate;
        for (var ip4 = 0; ip4 < this.prev_activations.lengthOfUnits; ip4++)
        {
            var a = this.prev_activations[ip4];

            var ip = ip4 * 4;
            this.dst_pxc_weithgs[ip + 0, ic4] = this.pxc_weithgs[ip + 0, ic4] - drate * a.xxxx;
            this.dst_pxc_weithgs[ip + 1, ic4] = this.pxc_weithgs[ip + 1, ic4] - drate * a.yyyy;
            this.dst_pxc_weithgs[ip + 2, ic4] = this.pxc_weithgs[ip + 2, ic4] - drate * a.zzzz;
            this.dst_pxc_weithgs[ip + 3, ic4] = this.pxc_weithgs[ip + 3, ic4] - drate * a.wwww;

            //this.dst_pxc_weithgs[ip, ic] = this.pxc_weithgs[ip, ic] - drate * this.prev_activations[ip];
        }
        //this.dst_pxc_weithgs[, ic4] = this.pxc_weithgs[, ic4] - drate;

        this.curr_ds[ic4] = d;
    }
}