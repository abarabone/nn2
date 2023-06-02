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

namespace nn.simd
{

    using number = Unity.Mathematics.float4;



    //struct aaa
    //{
    //    void aaaaa() => new NnActivations4<NnActivation4<number4>, number4>();
    //}

    [System.Serializable]
    public struct NnActivations : IDisposable
    {
        public NativeArray<number> currents;

        public int lengthOfNodes => this.currents.Length * 4;
        public int lengthOfUnits => this.currents.Length;

        //public IActivationValue<number4> this[int i]
        //{
        //    get => new NnActivation4() { value = this.currents[i] };
        //    set => this.currents[i] = value.Value;
        //}
        public number this[int i]
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
    //public interface IActivationValue<T> where T : struct
    //{
    //    public T Value { get; set; }
    //    //public void init(NativeArray<T> na, int i);
    //}
    //public struct NnActivation4 : IActivationValue<number4>
    //{
    //    public number4 value;
    //    public number4 Value
    //    {
    //        get => this.value;
    //        set => this.value = value;
    //    }
    //    //public void init(NativeArray<T> na, int i)
    //    //{
    //    //    this.currents = na;
    //    //}

    //    static public number4 operator *(NnActivation4 l, NnWeights4 r)
    //    {
    //        return 0;
    //    }
    //}


    [System.Serializable]
    public struct NnWeights : IDisposable
    {
        public NativeArray<number> c4xp;
        public int width4;

        public NativeArray<number> values => this.c4xp;//

        public int widthOfNodes => this.width4 * 4;
        public int widthOfUnits => this.width4;

        public int lengthOfUnits => this.c4xp.Length;
        public int lengthOfNodes => this.c4xp.Length * 4;

        public number this[int ic4, int ip]
        {
            get => this.c4xp[ip * this.widthOfUnits + ic4];
            set => this.c4xp[ip * this.widthOfUnits + ic4] = value;
        }

        public void Dispose()
        {
            if (!this.c4xp.IsCreated) return;

            this.c4xp.Dispose();
        }
    }





    public interface IActivationFunction
    {
        number Activate(number u);
        number Prime(number a);
        void InitWeights(NnWeights weights);
    }
    public struct ReLU : IActivationFunction
    {
        public number Activate(number u) => max(u, 0);
        public number Prime(number a) => sign(a);
        public void InitWeights(NnWeights weights) => weights.InitHe();
    }
    public struct Sigmoid : IActivationFunction
    {
        public number Activate(number u) => 1 / (1 + exp(-u));
        public number Prime(number a) => a * (1 - a);
        public void InitWeights(NnWeights weights) => weights.InitXivier();
    }
    public struct Affine4 : IActivationFunction
    {
        public number Activate(number u) => u;
        public number Prime(number a) => 1;
        public void InitWeights(NnWeights weights) => weights.InitRandom();
    }
}