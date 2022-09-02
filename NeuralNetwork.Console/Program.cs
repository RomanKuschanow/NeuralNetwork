using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var network = new Network(new List<int>() { 2, 2, 1 });

            var trainData = new Dictionary<List<double>, List<double>>()
            {
                { new List<double>() { 1, 1 },  new List<double>(){ 1 } },
                { new List<double>() { 1, 0 },  new List<double>(){ 1 } },
                { new List<double>() { 0, 1 },  new List<double>(){ 1 } },
                { new List<double>() { 0, 0 },  new List<double>(){ 0 } },
            };

            network.Train(trainData, 1000, 0.5);

            var oneOne = network.FeedForward(new List<double>() { 1, 1 })[0];
            var oneZero = network.FeedForward(new List<double>() { 1, 0 })[0];
            var zeroOne = network.FeedForward(new List<double>() { 0, 1 })[0];
            var zeroZero = network.FeedForward(new List<double>() { 0, 0 })[0];

            Console.WriteLine();

            Console.WriteLine(oneOne);
            Console.WriteLine(oneZero);
            Console.WriteLine(zeroOne);
            Console.WriteLine(zeroZero);
        }
    }
}
