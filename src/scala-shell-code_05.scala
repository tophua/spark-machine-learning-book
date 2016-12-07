/*
	This code is intended to be run in the Scala shell. 
	Launch the Scala Spark shell by running ./bin/spark-shell from the Spark directory.
	You can enter each line in the shell and see the result immediately.
	The expected output in the Spark console is presented as commented lines following the
	relevant code

	The Scala shell creates a SparkContex variable available to us as 'sc'
*/

// sed 1d train.tsv > train_noheader.tsv
// load raw data
val rawData = sc.textFile("/PATH/train_noheader.tsv")
val records = rawData.map(line => line.split("\t"))
records.first
// Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...

package shell1

def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
	DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
}
 
// investigate tree depth impact for Entropy impurity
val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Entropy)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.88%
10 tree depth, AUC = 76.26%
20 tree depth, AUC = 98.45%
*/

// investigate tree depth impact for Gini impurity
val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Gini)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.89%
10 tree depth, AUC = 78.37%
20 tree depth, AUC = 98.87%
*/

// investigate Naive Bayes parameters
def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
	val nb = new NaiveBayes
	nb.setLambda(lambda)
	nb.run(input)
}
val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainNBWithParams(dataNB, param)
	val scoreAndLabels = dataNB.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param lambda", metrics.areaUnderROC)
}
nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 lambda, AUC = 60.51%
0.01 lambda, AUC = 60.51%
0.1 lambda, AUC = 60.51%
1.0 lambda, AUC = 60.51%
10.0 lambda, AUC = 60.51%
*/

// illustrate cross-validation
// create a 60% / 40% train/test data split
val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
val train = trainTestSplit(0)
val test = trainTestSplit(1)
// now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
// in addition, we will evaluate the differing performance of regularization on training and test datasets
val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", test, model)
}
regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.480874%
0.001 L2 regularization parameter, AUC = 66.480874%
0.0025 L2 regularization parameter, AUC = 66.515027%
0.005 L2 regularization parameter, AUC = 66.515027%
0.01 L2 regularization parameter, AUC = 66.549180%
*/

// training set results
val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", train, model)
}
regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.260311%
0.001 L2 regularization parameter, AUC = 66.260311%
0.0025 L2 regularization parameter, AUC = 66.260311%
0.005 L2 regularization parameter, AUC = 66.238294%
0.01 L2 regularization parameter, AUC = 66.238294%
*/