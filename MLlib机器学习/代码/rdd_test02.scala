import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.util.KMeansDataGenerator
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.mllib.util.LogisticRegressionDataGenerator

object rdd_test02 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_test02")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 2.2节
    // 2.2.1 列统计汇总
    val data_path = "/home/huangmeiling/sample_stat.txt"
    val data = sc.textFile(data_path).map(_.split("\t")).map(f => f.map(f => f.toDouble))
    val data1 = data.map(f => Vectors.dense(f))
    val stat1 = Statistics.colStats(data1)
    stat1.max
    stat1.min
    stat1.mean
    stat1.variance
    stat1.normL1
    stat1.normL2

    // 2.2.2 相关系数
    val corr1 = Statistics.corr(data1, "pearson")
    val corr2 = Statistics.corr(data1, "spearman")
    val x1 = sc.parallelize(Array(1.0, 2.0, 3.0, 4.0))
    val y1 = sc.parallelize(Array(5.0, 6.0, 6.0, 6.0))
    val corr3 = Statistics.corr(x1, y1, "pearson")

    //2.2.3 卡方检验
    val v1 = Vectors.dense(43.0, 9.0)
    val v2 = Vectors.dense(44.0, 4.0)
    val c1 = Statistics.chiSqTest(v1, v2)

    // 2.3节
    // 2.3.2 生成样本
    val KMeansRDD = KMeansDataGenerator.generateKMeansRDD(sc, 40, 5, 3, 1.0, 2)
    KMeansRDD.count()
    KMeansRDD.take(5)

    val LinearRDD = LinearDataGenerator.generateLinearRDD(sc, 40, 3, 1.0, 2, 0.0)
    LinearRDD.count()
    LinearRDD.take(5)

    val LogisticRDD = LogisticRegressionDataGenerator.generateLogisticRDD(sc, 40, 3, 1.0, 2, 0.5)
    LogisticRDD.count()
    LogisticRDD.take(5)

  }

}