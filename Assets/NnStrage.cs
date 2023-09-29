using nn;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using NaughtyAttributes;
using Unity.Mathematics;

namespace nn.user
{
    //using Nnx = Nn<float, float, NnFloat, NnFloat.ForwardActivation, NnFloat.BackError, NnFloat.BackDelta>;
    using Nnx = Nn<float4, float, NnFloat4, NnFloat4.ForwardActivation, NnFloat4.BackError, NnFloat4.BackDelta>;

    public class NnStrage : MonoBehaviour
    {

        public bool isTextFile;

        public NnComponent nn;


        [Button]
        void abc()
        {
            if (this.isTextFile)
                this.writeToTextFile(this.nn.nn);
            else
                this.writeToFile(this.nn.nn);
        }

        void writeToFile(Nnx.NnLayers layers)
        {
            var fname = $"{Application.dataPath}/posing.nn";
            Debug.Log(fname);
            using var f = new FileStream(fname, FileMode.Create, FileAccess.Write);
            using var b = new BinaryWriter(f);

            var q =
                from l in layers.layers
                let ws = l.weights.values.Reinterpret<int>()
                let len = ws.Length
                from w in ws.Prepend(len)
                select w
                ;
            var bs = MemoryMarshal.AsBytes(q.ToArray().AsSpan());
            b.Write(bs);
        }

        void writeToTextFile(Nnx.NnLayers layers)
        {
            var fname = $"{Application.dataPath}/posing.txt";
            Debug.Log(fname);
            using var f = new StreamWriter(fname, false);

            foreach (var l in layers.layers)
            {
                var length = l.weights.values.Length * Nnx.Calc.NodesInUnit;
                var w = l.weights.widthOfUnit * Nnx.Calc.NodesInUnit;//Debug.Log($"{w} {l.weights.widthOfUnits} {Nnx.Calc.nodesInUnit}");
                var h = 0;
                if (w != 0) h = length / w;
                f.WriteLine($"{length} : {w} x {h}");

                if (w == 0) continue;

                var q = l.weights.values.Reinterpret<float>()
                    .Buffer(w)
                    ;

                foreach (var chunk in q)
                {
                    var line = string.Join(" ", from x in chunk select $"{x:00.00;-0.00}");

                    f.WriteLine(line);
                }
            }
        }
    }

}