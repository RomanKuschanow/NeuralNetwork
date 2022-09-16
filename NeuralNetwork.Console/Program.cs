using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var networkOR = new Network(new List<int>() { 2, 1 });
            var networkNAND = new Network(new List<int>() { 2, 1 });
            var networkAND = new Network(new List<int>() { 2, 1 });

            var OR = new Dictionary<List<decimal>, List<decimal>>()
            {
                { new List<decimal>() { 1, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 1, 0 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 0 },  new List<decimal>(){ 0 } },
            };

            var NAND = new Dictionary<List<decimal>, List<decimal>>()
            {
                { new List<decimal>() { 1, 1 },  new List<decimal>(){ 0 } },
                { new List<decimal>() { 1, 0 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 0 },  new List<decimal>(){ 1 } },
            };

            var AND = new Dictionary<List<decimal>, List<decimal>>()
            {
                { new List<decimal>() { 1, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 1, 0 },  new List<decimal>(){ 0 } },
                { new List<decimal>() { 0, 1 },  new List<decimal>(){ 0 } },
                { new List<decimal>() { 0, 0 },  new List<decimal>(){ 0 } },
            };

            var XOR = new Dictionary<List<decimal>, List<decimal>>()
            {
                { new List<decimal>() { 1, 1 },  new List<decimal>(){ 0 } },
                { new List<decimal>() { 1, 0 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 1 },  new List<decimal>(){ 1 } },
                { new List<decimal>() { 0, 0 },  new List<decimal>(){ 0 } },
            };

            networkOR.Train(OR, 5000, 10);

            var oneOne = networkOR.FeedForward(new List<decimal>() { 1, 1 }).Last()[0];
            var oneZero = networkOR.FeedForward(new List<decimal>() { 1, 0 }).Last()[0];
            var zeroOne = networkOR.FeedForward(new List<decimal>() { 0, 1 }).Last()[0];
            var zeroZero = networkOR.FeedForward(new List<decimal>() { 0, 0 }).Last()[0];

            Console.WriteLine();

            Console.WriteLine(oneOne);
            Console.WriteLine(oneZero);
            Console.WriteLine(zeroOne);
            Console.WriteLine(zeroZero);


            networkNAND.Train(NAND, 5000, 10);

            oneOne = networkNAND.FeedForward(new List<decimal>() { 1, 1 }).Last()[0];
            oneZero = networkNAND.FeedForward(new List<decimal>() { 1, 0 }).Last()[0];
            zeroOne = networkNAND.FeedForward(new List<decimal>() { 0, 1 }).Last()[0];
            zeroZero = networkNAND.FeedForward(new List<decimal>() { 0, 0 }).Last()[0];

            Console.WriteLine();

            Console.WriteLine(oneOne);
            Console.WriteLine(oneZero);
            Console.WriteLine(zeroOne);
            Console.WriteLine(zeroZero);


            networkAND.Train(AND, 5000, 10);

            oneOne = networkAND.FeedForward(new List<decimal>() 
            {
                networkOR.FeedForward(new List<decimal>() { 1, 1 }).Last()[0],
                networkNAND.FeedForward(new List<decimal>() { 1, 1 }).Last()[0] 
            }).Last()[0];

            oneZero = networkAND.FeedForward(new List<decimal>() 
            { 
                networkOR.FeedForward(new List<decimal>() { 1, 0 }).Last()[0],
                networkNAND.FeedForward(new List<decimal>() { 1, 0 }).Last()[0] 
            }).Last()[0];

            zeroOne = networkAND.FeedForward(new List<decimal>() 
            { 
                networkOR.FeedForward(new List<decimal>() { 0, 1 }).Last()[0],
                networkNAND.FeedForward(new List<decimal>() { 0, 1 }).Last()[0] 
            }).Last()[0];

            zeroZero = networkAND.FeedForward(new List<decimal>() 
            { 
                networkOR.FeedForward(new List<decimal>() { 0, 0 }).Last()[0],
                networkNAND.FeedForward(new List<decimal>() { 0, 0 }).Last()[0] 
            }).Last()[0];

            Console.WriteLine();

            Console.WriteLine(oneOne);
            Console.WriteLine(oneZero);
            Console.WriteLine(zeroOne);
            Console.WriteLine(zeroZero);

            Console.WriteLine();
            Console.WriteLine();

            var network = new Network(new List<int>() { 2, 2, 1 });

            network.Train(XOR, 5000, 10);

            oneOne = network.FeedForward(new List<decimal>() { 1, 1 }).Last()[0];
            oneZero = network.FeedForward(new List<decimal>() { 1, 0 }).Last()[0];
            zeroOne = network.FeedForward(new List<decimal>() { 0, 1 }).Last()[0];
            zeroZero = network.FeedForward(new List<decimal>() { 0, 0 }).Last()[0];

            Console.WriteLine();

            Console.WriteLine(oneOne);
            Console.WriteLine(oneZero);
            Console.WriteLine(zeroOne);
            Console.WriteLine(zeroZero);
        }
    }
}
