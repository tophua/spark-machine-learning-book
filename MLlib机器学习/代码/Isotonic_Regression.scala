
import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.mllib.regression.{ IsotonicRegression, IsotonicRegressionModel }

object Isotonic_Regression {

  def main(args: Array[String]) {
    // 构建Spark对象
    val conf = new SparkConf().setAppName("Isotonic_Regression")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 读取样本数据
    val data = sc.textFile("/home/jb-huangmeiling/sample_isotonic_regression_data.txt")
    val parsedData = data.map { line =>
      val parts = line.split(',').map(_.toDouble)
      (parts(0), parts(1), 1.0)
    }

    //样本数据划分训练样本与测试样本
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val tr = training.collect
    for (i <- 0 to tr.length - 1) {
      println(tr(i)._1 + "\t" + tr(i)._2)
    }
    val te = test.collect
    for (i <- 0 to te.length - 1) {
      println(te(i)._1 + "\t" + te(i)._2)
    }

    // 新建保序回归模型，并训练
    val model = new IsotonicRegression().setIsotonic(true).run(training)
    val x = model.boundaries
    val y = model.predictions
    println("boundaries" + "\t" + "predictions")
    for (i <- 0 to x.length - 1) {
      println(x(i) + "\t" + y(i))
    }

    // 误差计算
    val predictionAndLabel = test.map { point =>
      val predictedLabel = model.predict(point._2)
      (predictedLabel, point._2)
    }
    val print_predict = predictionAndLabel.collect
    println("prediction" + "\t" + "label")
    for (i <- 0 to print_predict.length - 1) {
      println(print_predict(i)._1 + "\t" + print_predict(i)._2)
    }
    val meanSquaredError = predictionAndLabel.map { case (p, l) => math.pow((p - l), 2) }.mean()
    println("Mean Squared Error = " + meanSquaredError)

    // 保存模型
    val ModelPath = "/user/huangmeiling/Isotonic_Regression_Model"
    model.save(sc, ModelPath)
    val sameModel = IsotonicRegressionModel.load(sc, ModelPath)
  }

}
