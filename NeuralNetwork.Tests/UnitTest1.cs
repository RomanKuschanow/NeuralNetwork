
using System;

namespace NeuralNetwork.Tests
{
    public class UnitTest1
    {
        [Fact]
        public void NeuralNetwork221Test()
        {
            //Arrange
            var sut = new Network(new List<int>() { 2, 2, 1 });

            //Act
            Func<double, int, double> normaliser = (x, i) => x - (i == 0 ? 50 : 150);
            var trainData = new Dictionary<List<double>, List<double>>()
            {
                { new List<double>() { 54.4, 165.1 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 1 } },
                { new List<double>() { 65.44, 183 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 0 } },
                { new List<double>() { 62.2, 178 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 0 } },
                { new List<double>() { 49, 152 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 1 } },
            };

            sut.Train(trainData, 1000, 0.5);

            var emily = sut.FeedForward(new List<double>() { 52.35, 160 }.Select((x, i) => normaliser(x, i)).ToList())[0];
            var frank = sut.FeedForward(new List<double>() { 63.4, 173 }.Select((x, i) => normaliser(x, i)).ToList())[0];

            //Assert
            emily.Should().BeGreaterThan(0.9);
            frank.Should().BeLessThan(0.1);

        }

        [Fact]
        public void NeuralNetworkXORTest()
        {
            //Arrange
            var sut = new Network(new List<int>() { 2, 2, 1 });

            //Act
            var trainData = new Dictionary<List<double>, List<double>>()
            {
                { new List<double>() { 1, 1 },  new List<double>(){ 0 } },
                { new List<double>() { 1, 0 },  new List<double>(){ 1 } },
                { new List<double>() { 0, 1 },  new List<double>(){ 1 } },
                { new List<double>() { 0, 0 },  new List<double>(){ 0 } },
            };

            sut.Train(trainData, 10000, 0.3);

            var oneOne = sut.FeedForward(new List<double>() { 1, 1 })[0];
            var oneZero = sut.FeedForward(new List<double>() { 1, 0 })[0];
            var zeroOne = sut.FeedForward(new List<double>() { 0, 1 })[0];
            var zeroZero = sut.FeedForward(new List<double>() { 0, 0 })[0];

            //Assert
            oneOne.Should().Be(0);
            oneZero.Should().Be(1);
            zeroOne.Should().Be(1);
            zeroZero.Should().Be(0);

        }

        [Fact]
        public void NeuralNetwork221ReverseTest()
        {
            //Arrange
            var sut = new Network(new List<int>() { 2, 2, 1 });

            //Act
            Func<double, int, double> normaliser = (x, i) => x - (i == 0 ? 50 : 150);
            var trainData = new Dictionary<List<double>, List<double>>()
            {
                { new List<double>() { 54.4, 165.1 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 1 } },
                { new List<double>() { 65.44, 183 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 0 } },
                { new List<double>() { 62.2, 178 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 0 } },
                { new List<double>() { 49, 152 }.Select((x, i) => normaliser(x, i)).ToList(),  new List<double>(){ 1 } },
            };

            sut.Train(trainData, 1000, 0.5);

            var res = sut.FeedBackward(new List<double>() { 1 });

            //Assert
        }
    }
}
