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

namespace nn.sisd
{

    using number = System.Single;




    public struct NnRandom
    {

        public Unity.Mathematics.Random rnd;


        public NnRandom(uint seed) =>
            this.rnd = new Unity.Mathematics.Random(seed);

        public number Next() =>
            (number)this.rnd.NextDouble(0, 1);

    }


    //public interface IActivationFunction<T> where T : unmanaged
    //{
    //    number Activate(T u);
    //    number Prime(T a);
    //    void InitWeights(NnWeights<T> weights);
    //}

    //public struct ReLU : IActivationFunction
    //{
    //    public number Activate(number u) => max(u, 0);
    //    public number Prime(number a) => sign(a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitHe();
    //}
    //public struct Sigmoid : IActivationFunction
    //{
    //    public number Activate(number u) => 1 / (1 + exp(-u));
    //    public number Prime(number a) => a * (1 - a);
    //    public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
    //}
    //public struct Affine : IActivationFunction
    //{
    //    public number Activate(number u) => u;
    //    public number Prime(number a) => 1;
    //    public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
    //}

}
