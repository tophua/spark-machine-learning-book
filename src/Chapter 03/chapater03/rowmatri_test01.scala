package chapater03

import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix

object rowmatri_test01 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[2]").setAppName("rowmatri_test01")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 3.6 分布式矩阵
    // 3.6.2 行矩阵（RowMatrix）
    val rdd1 = sc.parallelize(Array(Array(1.0, 2.0, 3.0, 4.0), Array(2.0, 3.0, 4.0, 5.0), Array(3.0, 4.0, 5.0, 6.0))).map(f => Vectors.dense(f))
    //分布式行矩阵
    val RM = new RowMatrix(rdd1)//参数是行
    val simic1 = RM.columnSimilarities(0.5)//计算每列之间的相似度
    val simic2 = RM.columnSimilarities()//计算每列之间的相似度
    val simic3 = RM.computeColumnSummaryStatistics()//计算每列的汇总统计
    simic3.max //最大值
    simic3.min//最小值
    simic3.mean//平均值
    val cc1 = RM.computeCovariance//计算每列之间的协方差
    val cc2 = RM.computeGramianMatrix//计算格拉姆矩阵
    val pc1 = RM.computePrincipalComponents(3)
    val svd = RM.computeSVD(4, true)//计算矩阵奇异分解
    val U = svd.U
    U.rows.foreach(println)
    val s = svd.s
    val V = svd.V
  }

}
