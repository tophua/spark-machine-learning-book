package chapter04
import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating

object als1 {

  //0 构建Spark对象
  val conf = new SparkConf().setAppName("ALS")
  val sc = new SparkContext(conf)
  Logger.getRootLogger.setLevel(Level.WARN)

  //1 读取样本数据
  val data = sc.textFile("ml-100k/test.data")
  val ratings = data.map(_.split(',') match {
    case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
  })

  //2 建立模型
  val rank = 10
  val numIterations = 20
  val model = ALS.train(ratings, rank, numIterations, 0.01)

  //3 预测结果
  val usersProducts = ratings.map {
    case Rating(user, product, rate) =>
      (user, product)
  }
  val predictions =
    model.predict(usersProducts).map {
      case Rating(user, product, rate) =>
        ((user, product), rate)//rate 预测分
    }
  val ratesAndPreds = ratings.map {
    case Rating(user, product, rate) =>
      ((user, product), rate)//rate实际评份
  }.join(predictions)
  /**用户评分误差**/  
  val MSE = ratesAndPreds.map {
    //user用户ID,product产品ID,r1实际评分,预测评分r2
    case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err //相乘
  }.mean()//平均值
  println("Mean Squared Error = " + MSE)

  //4 保存/加载模型
  model.save(sc, "myModelPath")
  val sameModel = MatrixFactorizationModel.load(sc, "myModelPath")

}
