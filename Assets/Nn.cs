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

//using nn;
//using numunit = System.Single;
//using number = System.Single;
//using nn;
//using numunit = System.Double;
//using number = System.Double;
using nn.simd;
using numunit = System.Single;
using number = Unity.Mathematics.float4;

public class Nn : MonoBehaviour
{

    public NnLayers nn;
    public int epoc;

    public int[] nodeList;
    public float learingRate;

    public ShowLayer[] show_layers;


    NnOtherAndLast<ReLU, Sigmoid> exe = new();
    //NnUinform<Sigmoid> exe = new();

    // Start is called before the first frame update
    void Start()
    {
        this.nn = new NnLayers(this.nodeList);

        this.exe.InitWeights(this.nn.layers);
    }



    JobHandle deps = default;

    // Update is called once per frame
    unsafe void Update()
    {
        //this.deps.Complete();
        //this.nn.switchWeightBuffers();
        this.deps = default;

        var ia = this.nn.layers.First().activations;
        var oa = this.nn.layers.Last().activations;

        var p = (numunit*)ia.currents.GetUnsafePtr();
        UnsafeUtility.MemClear(p, ia.lengthOfNodes * sizeof(numunit));
        var i = Unity.Mathematics.Random.CreateFromIndex((uint)Time.frameCount).NextInt(0, ia.lengthOfNodes);
        p[i] = 1;

        this.deps = this.exe.ExecuteForwardWithJob(this.nn.layers, this.deps);
        this.deps = this.exe.ExecuteBackwordWithJob(this.nn.layers, ia.currents, this.learingRate, this.deps);
        JobHandle.ScheduleBatchedJobs();
        //this.exe.ExecuteForward(this.nn.layers);
        //this.exe.ExecuteBackword(this.nn.layers, ia.activations, this.learingRate);

        this.deps.Complete();
        this.nn.switchWeightBuffers();
        this.epoc++;
    }

    unsafe void OnDisable()
    {
        this.deps.Complete();

        var ia = this.nn.layers.First().activations;
        var oa = this.nn.layers.Last().activations;
        logger($"i: ", ia.currents);
        logger($"o: ", oa.currents);

        //this.show_layers = this.nn.layers.toShow();
    }
    private void OnDestroy()
    {
        this.nn.Dispose();
    }

    void logger<T>(string desc, NativeArray<T> arr) where T:struct
    {
        var s = string.Join(" ", arr.Select(x => $"{x:f2}").Prepend(desc));
        Debug.Log(s);
    }
}


//public struct testjob : IJobParallelFor
//{
//    [ReadOnly]
//    public NnWeights srcs;

//    [ReadOnly]
//    //[WriteOnly]
//    public NnWeights dsts;

//    public void Execute(int i)
//    {
//        this.dsts[i, 0] = this.srcs[i, 0];
//    }
//}


//public struct NodeUnit
//{
//    public float a;

//}


[Serializable]
public class ShowLayer
{
    //[SerializeField]
    public number[] acts;

    //[SerializeField]
    public we[] weights;

    [Serializable]
    public class we
    {
        //[SerializeField]
        public number[] weights;
    }
}

//[Serializable]
//public class ShowLayer4
//{
//    //[SerializeField]
//    public number4[] acts;

//    //[SerializeField]
//    public we[] weights;

//    [Serializable]
//    public class we
//    {
//        //[SerializeField]
//        public number4[] weights;
//    }
//}

static public class ShowExtension
{
    static public ShowLayer[] toShow(this NnLayer[] layers)
    {
        var qa =
            from l in layers
            select l.activations.currents
            ;
        var qw =
            from l in layers
            select
                from w in l.weights.values.Chunks(l.weights.widthOfUnits)
                select w
            ;

        return qa.Zip(qw, (x, y) => new ShowLayer
        {
            acts = x.ToArray(),
            weights = y
                .Select(ws => new ShowLayer.we
                {
                    weights = ws.ToArray(),
                })
                .ToArray()
        })
        .ToArray();
    }
    //static public ShowLayer4[] toShow(this NnLayer4[] layers)
    //{
    //    var qa =
    //        from l in layers
    //        select l.activations.currents
    //        ;
    //    var qw =
    //        from l in layers
    //        select
    //            from w in l.weights.c4xp.Chunks(l.weights.length)
    //            select w
    //        ;

    //    return qa.Zip(qw, (x, y) => new ShowLayer4
    //    {
    //        acts = x
    //            .ToArray(),
    //        weights = y
    //            .Select(ws => new ShowLayer4.we
    //            {
    //                weights = ws.ToArray(),
    //            })
    //            .ToArray()
    //    })
    //    .ToArray();
    //}

    // 指定サイズのチャンクに分割する拡張メソッド
    public static IEnumerable<IEnumerable<T>> Chunks<T>
    (this IEnumerable<T> list, int size)
    {
        while (list.Any())
        {
            yield return list.Take(size);
            list = list.Skip(size);
        }
    }
}