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

namespace nn
{
    using number = System.Single;


    [System.Serializable]
    public struct NnActivations<T> : IDisposable
        where T : struct
    {
        public NativeArray<T> currents;

        //public int length => this.currents.Length;
        public int lengthOfNodes => this.currents.Length;
        public int lengthOfUnits => this.currents.Length;


        public T this[int i]
        {
            get => this.currents[i];
            set => this.currents[i] = value;
        }

        public void Dispose()
        {
            if (!this.currents.IsCreated) return;

            this.currents.Dispose();
        }
    }


    [System.Serializable]
    public struct NnWeights<T> : IDisposable
        where T : struct
    {
        public NativeArray<T> values;
        public int width;

        public int length => this.values.Length;
        public int lengthOfNodes => this.length;
        public int lengthOfUnits => this.length;

        public T this[int ix, int iy]
        {
            get => this.values[iy * this.width + ix];
            set => this.values[iy * this.width + ix] = value;
        }

        public void Dispose()
        {
            if (!this.values.IsCreated) return;

            this.values.Dispose();
        }
    }






    public interface IActivationFunction
    {
        number Activate(number u);
        number Prime(number a);
        void InitWeights(NnWeights<number> weights);
    }
    public struct ReLU : IActivationFunction
    {
        public number Activate(number u) => max(u, 0);
        public number Prime(number a) => sign(a);
        public void InitWeights(NnWeights<number> weights) => weights.InitHe();
    }
    public struct Sigmoid : IActivationFunction
    {
        public number Activate(number u) => 1 / (1 + exp(-u));
        public number Prime(number a) => a * (1 - a);
        public void InitWeights(NnWeights<number> weights) => weights.InitXivier();
    }
    public struct Affine : IActivationFunction
    {
        public number Activate(number u) => u;
        public number Prime(number a) => 1;
        public void InitWeights(NnWeights<number> weights) => weights.InitRandom();
    }

}