package chapater02

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
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    Logger.getRootLogger.setLevel(Level.WARN)

    val data_path = "/home/huangmeiling/sample_stat.txt"
    //转换
    val data = sc.textFile("ml-100k/sample_stat.txt").map(_.split("\t")).map(f => f.map(f => f.toDouble)) //转换成double类型
    //val data = sc.textFile(data_path)
    //将数据转换成 RDD[Vector]格式
    val data1 = data.map(f => Vectors.dense(f))
    val stat1 = Statistics.colStats(data1)
    //计算每列最大值,最小值,平均值,方差值,L1范数,L2范数
    stat1.max
    val max = stat1.max
    val min = stat1.min
    val men = stat1.mean
    val variance = stat1.variance
    val normL1 = stat1.normL1
    val normL2 = stat1.normL2
    println("最大值:%s,最小值:%s,平均值:%s,方差值:%s,L1范数:%s,L2范数:%s".format(max.toString(), min.toString(), men.toString(), variance.toString(), normL1.toString(), normL2.toString()))
    //Pearson相关是最常见的相关公式，用于计算连续数据的相关，比如计算班上学生数学成绩和语文成绩的相关可以用Pearson相关
    //spearman相关是专门用于分析顺序数据，就是那种只有顺序关系，但并非等距的数据，比如计算班上学生数学成绩排名和语文成绩排名的关系
    /**
     * 皮尔森相关系数:
     * 是用来反应两个变量相似程度的统计量。或者说可以用来计算两个向量的相似度（在基于向量空间模型的文本分类、用户喜好推荐系统中都有应用）
     * Spearman秩相关系数:
     * 
     */
    val corr1 = Statistics.corr(data1, "pearson")
    val corr2 = Statistics.corr(data1, "spearman")
    val x1 = sc.parallelize(Array(1.0, 2.0, 3.0, 4.0))
    val y1 = sc.parallelize(Array(5.0, 6.0, 6.0, 6.0))
    val corr3 = Statistics.corr(x1, y1, "pearson")//

    val v1 = Vectors.dense(43.0, 9.0)
    val v2 = Vectors.dense(44.0, 4.0)
    //检验
    val c1 = Statistics.chiSqTest(v1, v2)

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