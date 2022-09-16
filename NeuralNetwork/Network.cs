using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Network
    {
        private List<int> LayersPreset { get; }
        private List<List<List<KeyValuePair<int, int>>>> NeuronConnections { get; }
        public List<List<Neuron>> Neurons { get; }

        public Network(
            List<int> layersPreset,
            List<KeyValuePair<KeyValuePair<int, int>,KeyValuePair<decimal, int>>> weight = null, 
            List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, bool>>> bias = null, 
            bool haveBias = true
            ) 
            : this(layersPreset, layersPreset.Skip(1)
            .Select((l, i) => Enumerable.Range(0, l)
                .Select(n => Enumerable.Range(0, layersPreset[i])
                    .Select(c => new KeyValuePair<int, int>(i, c))
                    .ToList())
                .ToList())
            .ToList(), weight, bias, haveBias)
        { }

        public Network(
            List<int> layersPreset, 
            List<List<List<KeyValuePair<int, int>>>> neuronConnections, 
            List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, int>>> weight = null,
            List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, bool>>> bias = null, 
            bool haveBias = true
            )
        {
            LayersPreset = layersPreset;
            NeuronConnections = neuronConnections;
            Neurons = layersPreset.Skip(1).Select((l, i) => Enumerable.Range(0, l)
            .Select(n => new Neuron(NeuronConnections[i][n].Count, haveBias ? null : 0, haveBias))
            .ToList()).ToList();

            if (bias != null)
            {
                EditBias(bias);
            }
            if (weight != null)
            {
                EditWeight(weight);
            }
        }

        public void EditBias(List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, bool>>> bias)
        {
            foreach (var item in bias)
            {
                try
                {
                    Neurons[item.Key.Key][item.Key.Value].Bias = item.Value.Key;
                    Neurons[item.Key.Key][item.Key.Value].ChangeBias = item.Value.Value;
                }
                catch
                {
                    continue;
                }
            }
        }

        public void EditWeight(List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, int>>> weight)
        {
            foreach (var item in weight)
            {
                try
                {
                    Neurons[item.Key.Key][item.Key.Value].Weights[item.Value.Value] = item.Value.Key;
                }
                catch
                {
                    continue;
                }
            }
        }

        #region Train
        public void Train(Dictionary<List<decimal>, List<decimal>> trainData, int epochs, decimal learnRate = 0.1m)
        {
            for (int e = 0; e < epochs; e++)
            {
                foreach (var data in trainData)
                {
                    List<List<decimal>> pred = FeedForwardTrain(data.Key);
                    List<List<decimal>> ideal = new List<List<decimal>>() { data.Value };

                    //Console.WriteLine(Math.Round(MSELoss(data.Value, pred.Last()), 4));

                    for (int l = Neurons.Count - 1; l > -1; l--)
                    {
                        ideal.Insert(0, new List<decimal>());

                        for (int n = 0; n < Neurons[l].Count; n++)
                        {

                            List<decimal> inputs = NeuronConnections[l][n].Select(connetion => pred[connetion.Key][connetion.Value]).ToList();

                            Neurons[l][n].WeightsAndBiasUpdate(ideal[1].Select(i => 2 * (pred[l + 1][n] - i)).Sum(), inputs, learnRate);

                            ideal[0].Add(Neurons[l][n].SigmoidFeedForward(inputs));
                        }
                    }
                }
            }
        }

        private List<List<decimal>> FeedForwardTrain(List<decimal> inputs)
        {
            List<List<decimal>> data = new List<List<decimal>>() { inputs };

            for (int l = 0; l < Neurons.Count; l++)
            {
                data.Add(new List<decimal>());

                for (int n = 0; n < Neurons[l].Count; n++)
                {
                    data[l + 1].Add(Neurons[l][n].SigmoidFeedForward(NeuronConnections[l][n].Select(_n => data[_n.Key][_n.Value]).ToList()));
                }
            }

            return data;
        }
        #endregion

        public List<decimal> FeedForward(List<decimal> inputs)
        {
            List<List<decimal>> data = new List<List<decimal>>() { inputs };

            for (int l = 0; l < Neurons.Count; l++)
            {
                data.Add(new List<decimal>());

                for (int n = 0; n < Neurons[l].Count; n++)
                {
                    data[l + 1].Add(Neurons[l][n].SigmoidFeedForward(NeuronConnections[l][n].Select(_n => data[_n.Key][_n.Value]).ToList()));
                }
            }

            return data.Last();
        }

        public List<decimal> FeedBackward(List<decimal> inputs)
        {
            var neuronConnections = new List<List<List<KeyValuePair<int, int>>>>();

            for (int l = 0; l < NeuronConnections.Count; l++)
            {
                neuronConnections.Insert(0, new List<List<KeyValuePair<int, int>>>());

                for (int n = 0; n < LayersPreset[l]; n++)
                {
                    neuronConnections[0].Add(new List<KeyValuePair<int, int>>());
                }
            }

            for (int l = 0; l < NeuronConnections.Count; l++)
            {
                for (int n = 0; n < NeuronConnections[NeuronConnections.Count - 1 - l].Count; n++)
                {
                    for (int c = 0; c < NeuronConnections[NeuronConnections.Count - 1 - l][n].Count; c++)
                    {
                        neuronConnections[l][c].Add(new KeyValuePair<int, int>(l, n));
                    }
                }
            }

            var network = new Network(LayersPreset.Reverse<int>().ToList(), neuronConnections, haveBias: false);

            for (int l = 0; l < Neurons.Count; l++)
            {
                for (int n = 0; n < Neurons[NeuronConnections.Count - 1 - l].Count; n++)
                {
                    for (int w = 0; w < Neurons[NeuronConnections.Count - 1 - l][n].Weights.Count; w++)
                    {
                        network.EditWeight(new List<KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, int>>>()
                        {
                            new KeyValuePair<KeyValuePair<int, int>, KeyValuePair<decimal, int>>(new KeyValuePair<int, int>(l, n), new KeyValuePair<decimal, int>(Neurons[NeuronConnections.Count - 1 - l][n].Weights[w], w))
                        });
                    }
                }
            }

            return network.FeedForward(inputs);
        }

        public decimal MSELoss(List<decimal> yTrue, List<decimal> yPred)
        {
            if (yTrue.Count != yPred.Count)
            {
                throw new Exception("'yTrue' must have the same number of elements as 'yPred'");
            }

            return Math.Abs(Math.Round(yTrue.Select((i, c) => 2 * (yPred[c] - i)).Sum(), 5));
        }
    }
}
