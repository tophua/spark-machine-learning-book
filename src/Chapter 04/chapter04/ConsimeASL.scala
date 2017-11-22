package chapter04
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.util.StatCounter
import org.jblas.DoubleMatrix
/**
 * 余弦相似度,物品推荐
 */

object ConsimeASL {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    //评级数据
    val rawData = sc.textFile("ml-100k/u.data")
    //用户ID  | 影片ID   | 星级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    /*提取前三个字段即 用户ID  | 影片ID   | 星级 */
    val rawRatings = rawData.map(_.split("\t").take(3)) //取出前三个字段,第四列时间暂时不需要
    //Rating评级类参数对应用户ID,产品(即影片ID),实际星级
    //map方法将原来user ID,movie ID,星级的数组转换为对应的对象,从而创建所需的评级的数组集
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    val userPros = rawRatings.map { case Array(user, movie, rating) => (user.toInt, movie.toInt) }

    /**===================构建数据模型推荐=====================================**/
    //ratings.first()
    //构建训练推荐模型
    //参数说明:
    //rank:对应ALS模型中在低阶近似矩阵中的隐含特征个数,通常合理取值为10---200
    //iterations:对应运行时迭代次数,10次左右一般就挺好
    //lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况0.01,正则参数应该通过用非样本的测试数据进行交叉验证调整     
    val model = ALS.train(ratings, 50, 10, 0.01) //返回MatrFactorizationModel对象矩阵分解模型
    model.userFeatures //用户因子
    //model.userFeatures.count

    /***********物品推荐模型效果*******************/
    //从MovieLens 100K数据集生成相似电影

    //定义一个函数来计算两个向量之间的余弦相似度,
    def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
      //余弦相似度:两个向量的点积与各向量范数的乘积的商,相似度的取值在-1和1之间
      //相似度取值在-1和1之间,1表示完 全相似,0表示两者不相关(即无相似性)
      val retur = vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
      println(">>>>>>>>>>>>" + retur)
      retur
    }
    //以物品567为例从模型中取回其对应的因子,
    val itemId = 567
    //返回第一个数组而我们只需第一个值(实际上,数组里也只会有一个值,也就是该物品的因子向量)
    val itemFactor = model.productFeatures.lookup(itemId).head
    println("itemFactorSize:" + itemFactor.length)
    for (f <- itemFactor) {
      println("itemFactor:" + f)
    }
    val K = 10
    //创建一个DoubleMatrix对象,然后再用该对象来计算它与自己的相似度
    val itemVector = new DoubleMatrix(itemFactor)
    println("itemVector:" + itemVector.toString())
    cosineSimilarity(itemVector, itemVector) //测试余弦相似度,计算它与自己的相似度

    /***求各个物品的相似度与***/
    //现在求各个物品的余弦相似度
    val sims = model.productFeatures.map {
      case (id, factor) =>
        val factorVector = new DoubleMatrix(factor)
        val sim = cosineSimilarity(factorVector, itemVector)
        (id, sim)
    }
    //对物品按照相似度排序,然后取出与物品567最相似的前10个物品, 
    //top传入Ordering对象,它会告诉Spark根据键值对里的值排序(也就是用similarity排序)
    val sortedSims = sims.top(K)(Ordering.by { case (id, similarity) => similarity })
    println(sortedSims.mkString("\n"))

    /**
     * (567,1.0000000000000002)
     * (413,0.7431091494962073)
     * (1471,0.7301420755366541)
     * (288,0.7229307761535951)
     * (201,0.7174894985405192)
     * (895,0.7167458613072292)
     * (403,0.7129522148795585)
     * (219,0.712221807408798)
     * (670,0.7118086627261311)
     * (853,0.7014903255601453)
     */

    /*****电影数据*******/
    val movies = sc.textFile("ml-100k/u.item")
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    //获取前二列数据 ,从电影ID和标题,返回Map形式
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
    /**检查推荐的相似物品**/
    //对物品相似度排序,然后取出与物品567最相似的前10个物品
    //传入Ordering对象,它会告诉Spark根据键值对里的值排序(也就是用similarity排序)

    println(titles(itemId)) //Wes Craven's New Nightmare (1994)
    val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    //slice提取第1列开始到11列的特征矩阵
    sortedSims2.slice(1, 11).map { case (id, sim) => (titles(id), sim) }.mkString("\n" + ">>>>>")
    /**
     * (Hideaway (1995),0.6932331537649621)
     * (Body Snatchers (1993),0.6898690594544726)
     * (Evil Dead II (1987),0.6897964975027041)
     * (Alien: Resurrection (1997),0.6891221044611473)
     * (Stephen King's The Langoliers (1995),0.6864214133620066)
     * (Liar Liar (1997),0.6812075443259535)
     * (Tales from the Crypt Presents: Bordello of Blood (1996),0.6754663844488256)
     * (Army of Darkness (1993),0.6702643811753909)
     * (Mystery Science Theater 3000: The Movie (1996),0.6594872765176396)
     * (Scream (1996),0.6538249646863378)
     *
     */

    /**推荐模型效果的评估***/
    //找出用户789所接评价过的电影
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    //用户789找出第一个评级
    val actualRating = moviesForUser.take(1)(0) //
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
    //均方差
    val MSE = ratingsAndPredictions.map {
      case ((user, product), (actual, predicted)) => math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count
    //
    println("均方差:Mean Squared Error = " + MSE)
    val RMSE = math.sqrt(MSE) //均方根误差
    println("Root Mean Squared Error = " + RMSE)
    val actualMovies = moviesForUser.map(_.product)
    //算下给用户789推荐的前10个物品 
    val userId = 789
    val topKRecs = model.recommendProducts(userId, K) //参数是模型用户ID,num参数要推荐的物品个数 

    val predictedMovies = topKRecs.map(_.product)
    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()

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
    }

    //**MLib内置的评估函数**//*
    // MSE, RMSE and MAE
    import org.apache.spark.mllib.evaluation.RegressionMetrics
    /**
     * 标准误差(Standard error)为各测量值误差的平方和的平均值的平方根,
     * 故也称均方根误差(Root mean squared error)。在相同测量条件下进行的测量称为等精度测量
     */
    // next get all the movie ids per user, grouped by user id
    val userMovies = ratings.map { case Rating(user, product, rating) => (user, product) }.groupBy(_._1)
    //返回一个预测值与实际值
    val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
    //实例化一个RegressionMetrics对象需要一个键值对类型的RDD,其每一条记录对应每个数据点上相应的预测值与实际值
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    //均方差
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    //均方根误差
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
    // Mean Squared Error = 0.08231947642632852
    // Root Mean Squared Error = 0.2869137090247319

    // MAPK 准确率
    //RankingMetrics类用来计算基于排名的评估指标
    import org.apache.spark.mllib.evaluation.RankingMetrics
    //需要向我们之前的平均准确率函数传入一个键值对类型的RDD,
    //其键为给定用户预测的物品的ID数组,而值则是实际的物品ID数组
    val predictedAndTrueForRanking = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2)
        (predicted.toArray, actual.toArray)
    }
    //使用RankingMetrics类计算基于排名的评估指标,需要向我们之前平均率函数传入一个健值对类型的RDD
    //其键为给定用户预测的推荐物品的ID数组,而值则实际的物品ID数组
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    //平均正确率值 meanAveragePrecision
    println("平均准确率:Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
    // Mean Average Precision = 0.07171412913757183

    // Compare to our implementation, using K = 2000 to approximate the overall MAP
    //
    val MAPK2000 = allRecs.join(userMovies).map {
      case (userId, (predicted, actualWithIds)) =>
        val actual = actualWithIds.map(_._2).toSeq
        avgPrecisionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision = " + MAPK2000)
    // Mean Average Precision = 0.07171412913757186

  }
  def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
    val predK = predicted.take(k)
    var score = 0.0
    var numHits = 0.0
    //
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
