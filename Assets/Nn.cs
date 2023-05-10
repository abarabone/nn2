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

using number = System.Double;

public class Nn : MonoBehaviour
{

    public NnLayers nn;
    public int epoc;

    public int[] nodeList;
    public float learingRate;

    public ShowLayer[] show_layers;

    // Start is called before the first frame update
    void Start()
    {
        this.nn = new NnLayers(this.nodeList);

        //new testjob
        //{
        //    srcs = this.layers.layers[1].weights,
        //    dsts = this.layers.layers[2].weights,
        //}
        //.Schedule(10, 1)
        //.Complete();
    }

    // Update is called once per frame
    unsafe void Update()
    {
        var ia = this.nn.layers.First().activations;
        var oa = this.nn.layers.Last().activations;

        UnsafeUtility.MemClear(ia.activations.GetUnsafePtr(), ia.lengthOfNodes * sizeof(number));
        var i = Unity.Mathematics.Random.CreateFromIndex((uint)Time.frameCount).NextInt(0, ia.lengthOfNodes);
        ia[i] = 1;

        this.nn.ExecuteForward();
        this.nn.ExecuteBack(ia.activations, this.learingRate);

        this.epoc++;
    }

    unsafe void OnDisable()
    {
        var ia = this.nn.layers.First().activations;
        var oa = this.nn.layers.Last().activations;
        logger($"i: ", ia.activations);
        logger($"o: ", oa.activations);

        this.show_layers = this.nn.layers.toShow();
    }
    private void OnDestroy()
    {
        this.nn.Dispose();
    }

    void logger<T>(string desc, NativeArray<T> arr) where T:struct
    {
        var s = string.Join(" ", arr.Select(x => $"{x:f2}").Prepend(desc).SkipLast(1));
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


public struct NodeUnit
{
    public float a;

}


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

static public class ShowExtension
{
    static public ShowLayer[] toShow(this NnLayer[] layers)
    {
        var qa =
            from l in layers
            select l.activations.activations
            ;
        var qw =
            from l in layers
            select
                from w in l.weights.weights.Chunks(l.weights.widthWithBias)
                select w
            ;

        return qa.Zip(qw, (x, y) => new ShowLayer
        {
            acts = x
                .ToArray(),
            weights = y
                .Select(ws => new ShowLayer.we
                {
                    weights = ws.ToArray(),
                })
                .ToArray()
        })
        .ToArray();
    }

    // �w��T�C�Y�̃`�����N�ɕ�������g�����\�b�h
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