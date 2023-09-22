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

namespace nn
{


    [System.Serializable]
    public struct NnActivations<T> : IDisposable
        where T : unmanaged
    {

        public NativeArray<T> currents;


        public int lengthOfNodes { get; private set; }  // T ŠÖŒW‚È‚µ‚ÌŒÂ”
        public int lengthOfUnits { get; private set; }  // T ‚ÌŒÂ”@’[”‚ðœ‚­

        public int lengthOfFraction { get; private set; }   // ’[”


        public T this[int i]
        {
            get => this.currents[i];
            set => this.currents[i] = value;
        }


        unsafe NnActivations<T> alloc<TUnit>(int nodeLength, Allocator allocator = Allocator.TempJob)
            where TUnit : unmanaged
        {
            var nodesInUnit = sizeof(T) / sizeof(TUnit);

            this.lengthOfNodes = nodeLength;
            this.lengthOfUnits = nodeLength / nodesInUnit;
            this.lengthOfFraction = nodeLength - this.lengthOfUnits * nodesInUnit;

            var length = this.lengthOfNodes + (this.lengthOfFraction > 0 ? 1 : 0);

            var option = NativeArrayOptions.UninitializedMemory;
            this.currents = new NativeArray<T>(length, allocator, option);

            return this;
        }

        static public NnActivations<T> Create<TUnit>(int nodeLength, Allocator allocator = Allocator.TempJob)
            where TUnit : unmanaged
        {
            return new NnActivations<T>().alloc<TUnit>(nodeLength, allocator);
        }


        public NnActivations<T> CloneForTempJob()
        {
            var clone = this;

            var allocator = Allocator.TempJob;
            var option = NativeArrayOptions.UninitializedMemory;
            clone.currents = new NativeArray<T>(this.currents.Length, allocator, option);

            return clone;
        }


        public void Dispose()
        {
            if (!this.currents.IsCreated) return;

            this.currents.Dispose();
        }
    }

}
