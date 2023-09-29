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

namespace nn.user
{
    using nnfloat = NnFloat4;
    //using Nnx = Nn<float, float, NnFloat, NnFloat.ForwardActivation, NnFloat.BackError, NnFloat.BackDelta>;
    using Nnx = Nn<float4, float, NnFloat4, NnFloat4.ForwardActivation, NnFloat4.BackError, NnFloat4.BackDelta>;


    public class NnComponent : MonoBehaviour
    {

        public Nnx.NnLayers nn;
        public int epoc;

        public int[] nodeList;
        public float learingRate;

        //public ShowLayer[] show_layers;


        Nnx.NnOtherAndLast<NnFloat4.ReLU, NnFloat4.Sigmoid> exe = new();
        //Nnx.NnUinform<Sigmoid> exe = new();


        void Awake()
        {
            this.nn = new Nnx.NnLayers(this.nodeList);

            this.exe.InitWeights(this.nn.layers);
        }



        JobHandle deps = default;

        unsafe void Update()
        {
            this.deps.Complete();
            //this.nn.switchWeightBuffers();
            this.deps = default;

            //this.nn.AllocActivationWorks();
            this.nn.AllocDeltaWorks();
            var ia = this.nn.layers.First().activations;
            var oa = this.nn.layers.Last().activations;

            var p = (float*)ia.currents.GetUnsafeReadOnlyPtr();
            UnsafeUtility.MemClear(p, ia.currents.Length * sizeof(float));
            var i = Unity.Mathematics.Random.CreateFromIndex((uint)Time.frameCount).NextInt(0, ia.lengthOfNode);
            p[i] = 1;
            //logger($"i: ", ia.currents);
            //logger($"o: ", oa.currents);

            this.deps = this.exe.ExecuteForwardWithJob(this.nn.layers, this.deps);
            //this.deps.Complete();
            //logger($"i: ", ia.currents);
            //logger($"o: ", oa.currents);
            this.deps = this.exe.ExecuteBackwordWithJob(this.nn.layers, ia.currents, this.learingRate, this.deps);
            //this.deps.Complete();
            this.deps = this.nn.AddDeltaToWeightsWithDisposeTempJob(this.deps);
            JobHandle.ScheduleBatchedJobs();

            //this.deps.Complete();
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
            this.deps.Complete();

            this.nn.Dispose();
        }

        void logger<T>(string desc, NativeArray<T> arr) where T : struct
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


    //[Serializable]
    //public class ShowLayer
    //{
    //    //[SerializeField]
    //    public number[] acts;

    //    //[SerializeField]
    //    public we[] weights;

    //    [Serializable]
    //    public class we
    //    {
    //        //[SerializeField]
    //        public number[] weights;
    //    }
    //}

    ////[Serializable]
    ////public class ShowLayer4
    ////{
    ////    //[SerializeField]
    ////    public number4[] acts;

    ////    //[SerializeField]
    ////    public we[] weights;

    ////    [Serializable]
    ////    public class we
    ////    {
    ////        //[SerializeField]
    ////        public number4[] weights;
    ////    }
    ////}

    //static public class ShowExtension
    //{
    //    static public ShowLayer[] toShow(this NnLayer<number>[] layers)
    //    {
    //        var qa =
    //            from l in layers
    //            select l.activations.currents
    //            ;
    //        var qw =
    //            from l in layers
    //            select
    //                from w in l.weights.values.Chunks(l.weights.widthOfUnits)
    //                select w
    //            ;

    //        return qa.Zip(qw, (x, y) => new ShowLayer
    //        {
    //            acts = x.ToArray(),
    //            weights = y
    //                .Select(ws => new ShowLayer.we
    //                {
    //                    weights = ws.ToArray(),
    //                })
    //                .ToArray()
    //        })
    //        .ToArray();
    //    }
    //    //static public ShowLayer4[] toShow(this NnLayer4[] layers)
    //    //{
    //    //    var qa =
    //    //        from l in layers
    //    //        select l.activations.currents
    //    //        ;
    //    //    var qw =
    //    //        from l in layers
    //    //        select
    //    //            from w in l.weights.c4xp.Chunks(l.weights.length)
    //    //            select w
    //    //        ;

    //    //    return qa.Zip(qw, (x, y) => new ShowLayer4
    //    //    {
    //    //        acts = x
    //    //            .ToArray(),
    //    //        weights = y
    //    //            .Select(ws => new ShowLayer4.we
    //    //            {
    //    //                weights = ws.ToArray(),
    //    //            })
    //    //            .ToArray()
    //    //    })
    //    //    .ToArray();
    //    //}

    //    // �w��T�C�Y�̃`�����N�ɕ�������g�����\�b�h
    //    public static IEnumerable<IEnumerable<T>> Chunks<T>
    //    (this IEnumerable<T> list, int size)
    //    {
    //        while (list.Any())
    //        {
    //            yield return list.Take(size);
    //            list = list.Skip(size);
    //        }
    //    }
    //}
}