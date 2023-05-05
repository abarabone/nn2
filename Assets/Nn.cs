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

public class Nn : MonoBehaviour
{

    public NnLayers layers;
    public int epoc;

    public int[] nodeList;
    public float learingRate;

    public ShowLayer show_layers;

    // Start is called before the first frame update
    void Start()
    {
        this.layers = new NnLayers(this.nodeList);

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
        var ia = this.layers.layers.First().activations.activations;
        var oa = this.layers.layers.Last().activations.activations;

        UnsafeUtility.MemClear(ia.GetUnsafePtr(), ia.Length * sizeof(float));
        var i = Unity.Mathematics.Random.CreateFromIndex((uint)Time.frameCount).NextInt(0, ia.Length - 1);
        ia[i] = 1;

        this.layers.ExecuteForward();
        this.layers.ExecuteBack(ia, this.learingRate);

        this.epoc++;
    }

    private void OnDisable()
    {
        this.layers.ExecuteForward();

        var ia = this.layers.layers.First().activations.activations;
        var oa = this.layers.layers.Last().activations.activations;
        logger($"i: ", ia);
        logger($"o: ", oa);


    }
    private void OnDestroy()
    {
        this.layers.Dispose();
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
    [SerializeField]
    public float[] acts;
    [SerializeField]
    public we[] weights;

    public class we
    {
        public float[] weights;
    }
}