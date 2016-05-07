package chapter04
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.util.StatCounter

/**
 * A simple Spark app in Scala
 * 构建Spark推荐引擎
 */
object ScalaApp4 {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    //评级数据
    val rawData = sc.textFile("ml-100k/u.data")
    //用户ID  | 影片ID   | 星级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    /*提取前三个字段即 用户ID  | 影片ID   | 星级 */
    val rawRatings = rawData.map(_.split("\t").take(3))
    //Rating评级类参数对应用户ID,产品(即影片ID),实际星级
    //map方法将原来user ID,movie ID,星级的数组转换为对应的对象,从而创建所需的评级的数组集
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    /**===================构建数据模型推荐=====================================**/
    //ratings.first()
    //构建训练推荐模型
    //参数说明:rank对应ALS模型中的因子个数,通常合理取值为10---200
    //iterations:对应运行时迭代次数,10次左右一般就挺好
    //lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况0.01 
    val model = ALS.train(ratings, 50, 10, 0.01) //返回MatrFactorizationModel对象
    model.userFeatures //用户因子
    model.userFeatures.count
    model.productFeatures.count //物品因子 
    /**=================使用推荐模型==============================**/
    //该模型预测用户789对电影123的评级为3.12 
    val predictedRating = model.predict(789, 123) //方便地计算给定用户对给定物品的预期得分
    println("预测用户789对电影123的评级为:" + predictedRating)
    val userId = 789
    val K = 10
    //算下给用户789推荐的前10个物品 
    val topKRecs = model.recommendProducts(userId, K) //参数用户ID,num参数要推荐的物品个数 
    println("用户789推荐的前10个物品\t" + topKRecs.mkString("\n"))

    /*****检验推荐内容*******/
    val movies = sc.textFile("ml-100k/u.item")
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0

    //获取前二列数据 ,从电影ID和标题
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
    println("电影ID 123名称:" + titles(123))
    //找出用户789所接触过的电影
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    //查看这个用户评价了多少电影
    println("用户789评价了多少电影:" + moviesForUser.size)
    //用户789对电影做过评级进行降序排序,给出最高评级的前10部电影及名称
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
    //看一下前10个推荐
    println("=========看一下前10个推荐=============")
    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
    /************推荐评估模型效果*******************/
     //用户789找出第一个评级
    val actualRating = moviesForUser.take(1)(0)//
    //然后求模型的预计评级
    val predictedRatingR = model.predict(789, actualRating.product)
    //计算实际评级和预计评级的平方误差
    val squaredError = math.pow(predictedRatingR - actualRating.rating, 2.0)
    //首先从ratings提取用户和物品ID
    val usersProducts = ratings.map { case Rating(user, product, rating) => (user, product) }
    //对各个"用户-物品"对做预测,所得的RDD以"用户和物品ID"对作为主键,对应的预计评级作为值
    val predictions = model.predict(usersProducts).map {
      case Rating(user, product, rating) => ((user, product), rating)
    }
    //提出真实的评级,同时对ratingsRDD做映射以让"用户-物品"对为主键,实际评级为对应的值,就得到两个主键组织相同的RDD,
    //将两个连接起来,以创建一个新的RDD
    val ratingsAndPredictions = ratings.map {
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    //
    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    println("Mean Squared Error = " + MSE)
    val RMSE = math.sqrt(MSE)
    println("Root Mean Squared Error = " + RMSE)
    val actualMovies = moviesForUser.map(_.product)
    val predictedMovies = topKRecs.map(_.product)
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
    /*
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    val imBroadcast = sc.broadcast(itemMatrix)
    val allRecs = model.userFeatures.map {
      case (userId, array) =>
        val userVector = new DoubleMatrix(array)
        val scores = imBroadcast.value.mmul(userVector)
        val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
        val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
        (userId, recommendedIds)
    }*/
  }

  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    for ((p, i) <- predK.zipWithIndex) {
      if (actual.contains(p)) {
        numHits += 1.0
        score += numHits / (i.toDouble + 1.0)
      }
    }
    if (actual.isEmpty) {
      1.0
    } else {
      score / scala.math.min(actual.size, k).toDouble
    }
  }

}
