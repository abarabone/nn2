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
using static UnityEngine.UI.CanvasScaler;

namespace nn
{

    /// <summary>
    /// p   ... previous
    /// c   ... current
    /// n   ... next
    /// _x_ ... かける（横列×縦行）
    /// _1  ... T を構成するユニット１個
    /// _n  ... T を構成するユニットｎ個（ T そのもの）
    /// </summary>
    [System.Serializable]
    public struct NnWeights<T> : IDisposable where T : unmanaged
    {

        NativeArray<T> cn_x_p1;
        int width_n;
        //int unitLength;

        public NativeArray<T> values => this.cn_x_p1;

        //public int widthOfNodes => this.width_n * this.unitLength;
        public int widthOfUnits => this.width_n;

        //public int lengthOfNodes => this.cn_x_p1.Length * this.unitLength;
        public int lengthOfUnits => this.cn_x_p1.Length;

        public T this[int ic_n, int ip_1]
        {
            get => this.cn_x_p1[ip_1 * this.widthOfUnits + ic_n];
            set => this.cn_x_p1[ip_1 * this.widthOfUnits + ic_n] = value;
        }
        public T this[int i_n]
        {
            get => this.cn_x_p1[i_n];
            set => this.cn_x_p1[i_n] = value;
        }


        //public unsafe NnWeights<T> Alloc(NnActivations<T> prev, NnActivations<T> curr)
        public unsafe NnWeights<T> Alloc<T1>(int prevNodeLength, int currNodeLength)
            where T1 : unmanaged
        {
            var nodesInUnit = sizeof(T) / sizeof(T1);
            var baseWidth = currNodeLength / nodesInUnit;

            var weightWidth = baseWidth + (currNodeLength - baseWidth > 0 ? 1 : 0);
            var weightHeight = prevNodeLength + 1;// +1 : bias
            var weightLength = weightWidth * weightHeight;

            var allocator = Allocator.Persistent;
            var option = NativeArrayOptions.UninitializedMemory;
            this.cn_x_p1 = new NativeArray<T>(weightLength, allocator, option);
            this.width_n = weightWidth;

            return this;
        }

        static public NnWeights<T> Create<T1>(int prevNodeLength, int currNodeLength)
            where T1 : unmanaged
        {
            return new NnWeights<T>().Alloc<T1>(prevNodeLength, currNodeLength);
        }


        public NnWeights<T> CloneForTempJob()
        {
            var clone = this;

            var allocator = Allocator.TempJob;
            var option = NativeArrayOptions.UninitializedMemory;
            this.cn_x_p1 = new NativeArray<T>(this.cn_x_p1.Length, allocator, option);

            return clone;
        }

        public void Dispose()
        {
            if (!this.cn_x_p1.IsCreated) return;

            this.cn_x_p1.Dispose();
        }
    }

}
