package chapter07

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.rdd.RDD
/**
 * 聚类
 */
object ScalaApp7 {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    /**电影数据***/
    val movies = sc.textFile("ml-100k/u.item")
    movies.first()
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    //电影题材 
    val genres = sc.textFile("ml-100k/u.genre")
    genres.take(5).foreach(println)
    /**
     * 数字表示相关题材的索引,索引对应了每部电影关于题材
     * unknown|0
     * Action|1
     * Adventure|2
     * Animation|3
     * Children's|4
     */
    //对电影题材料每一行进行分隔,得到具体的<索引,题材>键值对
    //处理最后空,Map列进行调换(array(1), array(0))
    val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1), array(0))).collectAsMap
    //genreMap:Map[String,String] = Map(2 -> Adventure, 5 -> Comedy, 12 -> Musical, 15 -> Sci-Fi, 
    //8 -> Drama, 18 -> Western, 7 -> Documentary, 17 -> War, 1 -> Action, 4 -> Children's, 11 -> Horror, 14 -> Romance,
    //6 -> Crime, 0 -> unknown, 9 -> Fantasy, 16 -> Thriller, 3 -> Animation, 10 -> Film-Noir, 13 ->Mystery)
    println(genreMap)
    /**
     * 电影数据和题材映射关系创建新的RDD,其中包含电影ID,标题和题材
     */
    val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
      //slice截取数据,提取第5列开始到最后1列      
      val genres = array.toSeq.slice(5, array.size)
      //WrappedArray(0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0)
      println("genres>>>>>>>>>>>" + genres)
      println("zipWithIndex>>>>>>>>>>>" + genres.zipWithIndex)
      //zipWithIndex方法统计包含题材索引的集合(0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
      val genresAssigned = genres.zipWithIndex.filter {
        //idx索引,过虑掉等于1
        case (g, idx) =>
          g == "1"
      }.map {
        //将集合中的索引映射到对应的文件信息
        case (g, idx) =>
          println(idx + ">>>>>" + genreMap(idx.toString))
          genreMap(idx.toString)
      }
      (array(0).toInt, (array(1), genresAssigned))
    }
    println(titlesAndGenres.first)
    // 电影ID,标题和题材
    // (1,(Toy Story (1995),ArrayBuffer(Animation, Children's, Comedy)))
    /***训推荐模型***/
    import org.apache.spark.mllib.recommendation.ALS
    import org.apache.spark.mllib.recommendation.Rating
    //获取用户和电影的向量
    val rawData = sc.textFile("ml-100k/u.data")
    //用户ID  | 影片ID   | 星级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    /*提取前三个字段即 用户ID  | 影片ID   | 星级 */
    val rawRatings = rawData.map(_.split("\t").take(3))
    //Rating评级类参数对应用户ID,产品(即影片ID),实际星级
    //map方法将原来user ID,movie ID,星级的数组转换为对应的对象,从而创建所需的评级的数组集
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    //构建训练推荐模型最小二乘法    
    //参数说明:rank对应ALS模型中的因子个数,通常合理取值为10---200
    //iterations:对应运行时迭代次数,10次左右一般就挺好
    //lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况0.01 
    val alsModel = ALS.train(ratings, 50, 10, 0.1) //返回了两个键值RDD(userFeatures和productFeatures)
    alsModel.userFeatures //这为用户ID
    alsModel.productFeatures //这个电影ID,值为相关因素

    // extract factor vectors 
    import org.apache.spark.mllib.linalg.Vectors
    //电影ID,值为相关因素
    val movieFactors = alsModel.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val movieVectors = movieFactors.map(_._2) //提取相关因素
    //用户ID,值为相关因素
    val userFactors = alsModel.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val userVectors = userFactors.map(_._2) //提取相关因素

    // investigate distribution of features   
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    //相关因素特征向量的分布,RowMatrix进行各种统计
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    println("Movie factors mean: " + movieMatrixSummary.mean)
    println("Movie factors variance: " + movieMatrixSummary.variance)
    println("User factors mean: " + userMatrixSummary.mean)
    println("User factors variance: " + userMatrixSummary.variance)
    // Movie factors mean: [0.28047737659519767,0.26886479057520024,0.2935579964446398,0.27821738264113755, ... 
    // Movie factors variance: [0.038242041794064895,0.03742229118854288,0.044116961097355877,0.057116244055791986, ...
    // User factors mean: [0.2043520841572601,0.22135773814655782,0.2149706318418221,0.23647602029329481, ...
    // User factors variance: [0.037749421148850396,0.02831191551960241,0.032831876953314174,0.036775110657850954, ...

    /***训练聚类模型***/
    // run K-means model on movie factor vectors
    import org.apache.spark.mllib.clustering.KMeans
    val numClusters = 5 //设置K
    val numIterations = 10 //最大迭代次数
    val numRuns = 3 //训练次数
    //注意聚类不需要标签,所以不用LablePoint实例
    val movieClusterModel = KMeans.train(movieVectors, numClusters, numIterations, numRuns)
    //更改最大迭代次数
    val movieClusterModelConverged = KMeans.train(movieVectors, numClusters, 100)

    /**使用聚类模型进行预测**/
    // train user model
    val userClusterModel = KMeans.train(userVectors, numClusters, numIterations, numRuns)
    // predict a movie cluster for movie 1
    val movie1 = movieVectors.first
    val movieCluster = movieClusterModel.predict(movie1)
    println(movieCluster)
    // 4
    // predict clusters for all movies
    val predictions = movieClusterModel.predict(movieVectors)
    println(predictions.take(10).mkString(","))
    // 0,0,1,1,2,1,0,1,1,1
    import breeze.linalg._
    import breeze.numerics.pow
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum
    // join titles with the factor vectors, and compute the distance of each vector from the assigned cluster center
    val titlesWithFactors = titlesAndGenres.join(movieFactors)
    val moviesAssigned = titlesWithFactors.map {
      case (id, ((title, genres), vector)) =>
        val pred = movieClusterModel.predict(vector)
        val clusterCentre = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
        (id, title, genres.mkString(" "), pred, dist)
    }
    val clusterAssignments = moviesAssigned.groupBy { case (id, title, genres, cluster, dist) => cluster }.collectAsMap

    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k:")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map { case (_, title, genres, _, d) => (title, genres, d) }.mkString("\n"))
      println("=====\n")
    }
    // clustering mathematical evaluation

    // compute the cost (WCSS) on for movie and user clustering
    val movieCost = movieClusterModel.computeCost(movieVectors)
    val userCost = userClusterModel.computeCost(userVectors)
    println("WCSS for movies: " + movieCost)
    println("WCSS for users: " + userCost)
    // WCSS for movies: 2586.0777166339426
    // WCSS for users: 1403.4137493396831

    // cross-validation for movie clusters
    val trainTestSplitMovies = movieVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainMovies = trainTestSplitMovies(0)
    val testMovies = trainTestSplitMovies(1)
    val costsMovies = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainMovies, numIterations, k, numRuns).computeCost(testMovies)) }
    println("Movie clustering cross-validation:")
    costsMovies.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
    /*
    Movie clustering cross-validation:
    WCSS for K=2 id 942.06
    WCSS for K=3 id 942.67
    WCSS for K=4 id 950.35
    WCSS for K=5 id 948.20
    WCSS for K=10 id 943.26
    WCSS for K=20 id 947.10
    */

    // cross-validation for user clusters
    val trainTestSplitUsers = userVectors.randomSplit(Array(0.6, 0.4), 123)
    val trainUsers = trainTestSplitUsers(0)
    val testUsers = trainTestSplitUsers(1)
    val costsUsers = Seq(2, 3, 4, 5, 10, 20).map { k => (k, KMeans.train(trainUsers, numIterations, k, numRuns).computeCost(testUsers)) }
    println("User clustering cross-validation:")
    costsUsers.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }
    /*
    User clustering cross-validation:
    WCSS for K=2 id 544.02
    WCSS for K=3 id 542.18
    WCSS for K=4 id 542.38
    WCSS for K=5 id 542.33
    WCSS for K=10 id 539.68
    WCSS for K=20 id 541.21
	*/
  }
}