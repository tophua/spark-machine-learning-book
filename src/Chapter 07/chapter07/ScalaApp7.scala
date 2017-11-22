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
    println(movies.first())
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    //电影题材 
    val genres = sc.textFile("ml-100k/u.genre")
    genres.take(5).foreach(println)
    /**
     * 数字表示相关题材的索引,索引对应了每部电影关于题材
     *  unknown|0  未知
     *  Action|1   动作
     *  Adventure|2  冒险
     *  Animation|3  动画片
     *  Children's|4 儿童片
     *  Comedy|5     喜剧
     *  Crime|6      犯罪片
     *  Documentary|7  记录
     *  Drama|8        剧情
     *  Fantasy|9      幻想
     *  Film-Noir|10   黑色电影
     *  Horror|11      恐怖
     *  Musical|12     音乐
     *  Mystery|13     悬念
     *  Romance|14		  爱情
     *  Sci-Fi|15      科幻
     *  Thriller|16    惊悚
     *  War|17         战争
     *  Western|18     西部
     */
    //对电影题材料每一行进行分隔,得到具体的<索引,题材>键值对
    //处理最后空,Map列进行调换(array(1), array(0))
    val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1), array(0))).collectAsMap()
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
      //println("genres>>>>>>>>>>>" + genres)
         //zipWithIndex方法统计包含题材索引的集合(0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
      //println("zipWithIndex>>>>>>>>>>>" + genres.zipWithIndex) //返回对偶列表，第二个组成部分是元素下标 ,从0开始
      //zipWithIndex ArrayBuffer((0,0), (0,1), (0,2), (1,3), (1,4), (1,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12), (0,13), (0,14), (0,15), (0,16), (0,17), (0,18))
      val genresAssigned = genres.zipWithIndex.filter {
        //Key值过虑掉等于1,idx是value值
        case (g, idx) =>
         //println("g:"+g)
          g == "1"          
      }.map {
        //将集合中的索引映射到对应的文件信息
        case (g, idx) =>
          //println(idx + ">>>>>" + genreMap(idx.toString))
          genreMap(idx.toString)
      }
      //genresAssigned ArrayBuffer(Animation, Children's, Comedy)
      //电影ID,标题和题材
      (array(0).toInt, (array(1), genresAssigned))
    }
    //
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
    println(rawRatings.toArray())
    //Rating评级类参数对应用户ID,产品(即影片ID),实际星级
    //map方法将原来user ID,movie ID,星级的数组转换为对应的对象,从而创建所需的评级的数组集
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    /**获取用户和电影的因素向量,需要训练一个新的推荐模型**/   
    //参数说明:rank对应ALS模型中的因子个数,通常合理取值为10---200
    //iterations:对应运行时迭代次数,10次左右一般就挺好
    //lambda:该参数控制模型的正则化过程,从而控制模型的过拟合情况0.01     
    val alsModel = ALS.train(ratings, 50, 10, 0.1) //返回了两个键值RDD(userFeatures和productFeatures)
    alsModel.userFeatures //这为用户ID和因子
    alsModel.productFeatures //这个电影ID,值为相关因素

    //提取相关的因素并转化到MLlib的Vector中作为聚类模型的训练输入
    import org.apache.spark.mllib.linalg.Vectors
    //电影ID,值为相关因素Vectors
    val movieFactors = alsModel.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val movieVectors = movieFactors.map(_._2) //提取电影相关因素
    //用户ID,值为相关因素
    val userFactors = alsModel.userFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val userVectors = userFactors.map(_._2) //提取用户相关因素

    // investigate distribution of features   
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    //相关因素特征向量的分布,RowMatrix进行各种统计
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    println("Movie factors mean: " + movieMatrixSummary.mean)//平均值
    println("Movie factors variance: " + movieMatrixSummary.variance)//方差
    println("User factors mean: " + userMatrixSummary.mean)//平均值
    println("User factors variance: " + userMatrixSummary.variance)//方差
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
    //[0.033130496740341187,0.26705870032310486,-0.06470431387424469,-0.27620309591293335,-0.5579972863197327,-0.36105620861053467,0.2515413761138916,0.503909707069397,-0.18813224136829376,0.19052250683307648,-0.12899400293827057,-0.34381452202796936,-0.05027823895215988,-0.03820587322115898,0.3110596537590027,0.05205772444605827,0.2201911360025406,-0.3292890787124634,0.3878093957901001,0.3220963180065155,-0.34360307455062866,-0.3125258982181549,-0.03509145975112915,0.04936598241329193,0.050676699727773666,0.2289056032896042,-0.13943906128406525,0.0679071769118309,-0.178903266787529,0.16504788398742676,0.6243731379508972,-0.35920676589012146,-0.07185130566358566,0.357896089553833,-0.14195860922336578,-0.17070387303829193,-0.18561360239982605,0.0018937239656224847,0.1121269091963768,0.11422386020421982,0.28339502215385437,-0.2074926495552063,-0.3413117527961731,0.11888174712657928,0.19513817131519318,-0.18359383940696716,-0.0384325347840786,0.018584446981549263,-0.0346914604306221,-0.16087883710861206]
    //println(">>>>>>>>>>>>"+movie1)
    //单独样子预测
    val movieCluster = movieClusterModel.predict(movie1)
    println(movieCluster)
    // 聚类预测  4
    // predict clusters for all movies,传入一个RDD数组对多个输入样本进行预测
    val predictions = movieClusterModel.predict(movieVectors)
    println(predictions.take(10).mkString(","))
    // 0,0,1,1,2,1,0,1,1,1   
    
    /**=======================**/
    import breeze.linalg._
    import breeze.numerics.pow
    //计算类中心的距离,定义欧拉距离之和,矩阵平方差之和,v1中心点,v2预测值
    //公式:计算每个类簇中样本与类中心的平方差,并最后求和,等价于将每个样本分配到欧拉距离最近的类中心
    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]): Double = pow(v1 - v2, 2).sum //平方差之和
    // join titles with the factor vectors, and compute the distance of each vector from the assigned cluster center
    // movieFactors:电影ID,值为相关因素Vectors titlesAndGenres:电影ID,标题和题材   
    val titlesWithFactors = titlesAndGenres.join(movieFactors)
    //org.apache.spark.rdd.RDD[(Int, ((String, Seq[String]),Vector))]
    /**
     * Array((1084,((Anne Frank Remembered (1995),ArrayBuffer(Documentary)),[0.216
      24723076820374,0.4357949495315552,0.038032110780477524,-0.5478736758232117,0.084
      12806689739227,-0.29648417234420776,0.28614675998687744,0.0624297633767128,-0.28
      72799038887024,0.05472242459654808,0.5983230471611023,0.40144190192222595,0.1740
      2538657188416,-0.6979153752326965,0.12716177105903625,-0.7612598538398743,0.0020
      523574203252792,0.42382073402404785,-0.27595090866088867,0.2634444236755371,0.52
      92174220085144,0.36286401748657227,0.2620362937450409,-0.21733589470386505,0.019
      550463184714317,0.19473381340503693,0.34018516540527344,-0.13918174803256989,0.5
      856925845146179,0.41255903244018555,-0.26579615473747253,-0.00452867476269602...
     */
    val moviesAssigned = titlesWithFactors.map {
      //id 电影ID,title 电影标题,genres 电影题材分类,vector 预测样本向量
      case (id, ((title, genres), vector)) =>
        println("id:"+id+"\t title:"+title+"\t genres:"+genres+"\t vector:"+vector)
        //预测聚类vector
        val pred = movieClusterModel.predict(vector)
        //根据预测分类,返回聚类中心点
        val clusterCentre = movieClusterModel.clusterCenters(pred)
      //计算 类中心的距离
        val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))
        //id 电影ID,title 电影标题,genres 电影题材分类,cluster 类别索引,dist 类中心的距离
        //id:384   title:Naked Gun 33 1/3: The Final Insult (1994)  genres:Comedy cluster:4 dist:1.4373933940893153
        (id, title, genres.mkString(" "), pred, dist)
    }
    //clusterAssignments 键是类簇的标识,值是电影ID,标题,题材分类,类别索引及类中心的距离
    val clusterAssignments = moviesAssigned.groupBy { 
         //id 电影ID,title 电影标题,genres 电影题材分类,cluster 类别索引,dist 类中心的距离
          case (id, title, genres, cluster, dist) => 
            //id:384   title:Naked Gun 33 1/3: The Final Insult (1994)  genres:Comedy cluster:4 dist:1.4373933940893153           
            cluster 
          }.collectAsMap //collectAsMap 返回hashMap包含所有RDD中的分片，key如果重复，后边的元素会覆盖前面的元素
     
      /***
       *Map(2 -> CompactBuffer((752,Replacement Killers, The (1998),A
       * ction Thriller,2,1.9724203006574639), (996,Big Green, The (1995),Children's Come
       * dy,2,1.322760647925611), (520,Great Escape, The (1963),Adventure War,2,0.8597175
       * 76853882), (204,Back to the Future (1985),Comedy Sci-Fi,2,0.7159828491568067), (
       * 228,Star Trek: The Wrath of Khan (1982),Action Adventure Sci-Fi,2,1.388725653881
       * 7503), (628,Sleepers (1996),Crime Drama,2,1.1310470417515681), (452,Jaws 2 (1978
       * ),Action Horror,2,1.800937758377417), (1168,Little Buddha (1993),Drama,2,1.06190
       * 7444051839), (1480,Herbie Rides Again (1974),Adventure Children's Comedy,2,1.456
       * 436949842105), (808,Program, The (1993),Action Drama,2,0.8748734019728581), (109
       * 6,Commandments (1997),Romance,2,0.3643293590048631), (696,City Hall (1996),Drama
       *  Thriller,2,0.9579166240468663), (196,Dead Poets Society (1989),Drama,2,0.636856
       * 9358036571), (96,Terminator 2: Judgment Day (1991),Action Sci-Fi Thriller,2,1.25
       * 80047660555096), (216,When Harry Met Sally... (1989),Comedy Romance,2,0.70033609
       * 13070389), (1664,8 Heads in a Duffel Bag (1997),Comedy,2,2.4554629438409545), (1
       * 44,Die Hard (1988),Action Thriller,2,1.3412361022113162), (84,Robert A. Heinlein
       * 's The Puppet Masters (1994),Horror Sci-Fi,2,1.9030740381319178), (76,Carlito's
       * Way (1993),Crime Drama,2,1.1133138026316973)       
       */
       println("clusterAssignments:"+clusterAssignments)
    //枚举每个类簇并输出距离类中心最近的前20部电影,以类别索引升序排序
    for ((k, v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      //
      println(s"Cluster $k:")
      //id 电影ID,title 电影标题,genres 电影题材分类,cluster 类别索引,dist 类中心的距离
      val m = v.toSeq.sortBy(_._5)//以类中心的距离升序排序
      println(m.take(20).map { case (_, title, genres, _, d) => (title, genres, d) }.mkString("\n"))
      println("=====\n")
    }
    /**
     * Cluster 0:
        (Machine, The (1994),Comedy Horror,0.04719537328883158)
        (Amityville 1992: It's About Time (1992),Horror,0.08210955059973332)
        (Amityville: A New Generation (1993),Horror,0.08210955059973332)
        (Amityville: Dollhouse (1996),Horror,0.09105596052479828)
        (Gordy (1995),Comedy,0.09229835886615274)
        (Venice/Venice (1992),Drama,0.09856910616770778)
        (Somebody to Love (1994),Drama,0.09964665321683677)
        (Boys in Venice (1996),Drama,0.09964665321683677)
        (Johnny 100 Pesos (1993),Action Drama,0.10830395926470877)
        (Falling in Love Again (1980),Comedy,0.11473033718000594)
        (Coldblooded (1995),Action,0.11965821378272547)
        (Babyfever (1994),Comedy Drama,0.12029901912371525)
        (Beyond Bedlam (1993),Drama Horror,0.12821241487193066)
        (Mighty, The (1998),Drama,0.13162481821279456)
        (War at Home, The (1996),Drama,0.1320111841926296)
        (Getting Away With Murder (1996),Comedy,0.1350664524646343)
        (Police Story 4: Project S (Chao ji ji hua) (1993),Action,0.13590358256026017)
        (Catwalk (1995),Documentary,0.13863148533098932)
        (Small Faces (1995),Drama,0.13863452630085718)
        (Homage (1995),Drama,0.1386687227413549)
        =====
        
        Cluster 1:
        (Last Time I Saw Paris, The (1954),Drama,0.14368041720283634)
        (Witness (1985),Drama Romance Thriller,0.20289573321868362)
        (Substance of Fire, The (1996),Drama,0.22290536638232972)
        (Beans of Egypt, Maine, The (1994),Drama,0.25529669455616094)
        (Mamma Roma (1962),Drama,0.2807936393846296)
        (Wife, The (1995),Comedy Drama,0.34170271990636986)
        (They Made Me a Criminal (1939),Crime Drama,0.347143420044582)
        (Angel and the Badman (1947),Western,0.35400733039995097)
        (Sleepover (1995),Comedy Drama,0.35484780272008015)
        (Love Is All There Is (1996),Comedy Drama,0.35484780272008015)
        (Century (1993),Drama,0.35484780272008015)
        (Spellbound (1945),Mystery Romance Thriller,0.37047839948014666)
        (Casablanca (1942),Drama Romance War,0.38266779087317565)
        (Cosi (1996),Comedy,0.3838952037901038)
        (Nelly & Monsieur Arnaud (1995),Drama,0.3889506830636553)
        (Object of My Affection, The (1998),Comedy Romance,0.4009730742531602)
        (Vertigo (1958),Mystery Thriller,0.4033444666147566)
        (An Unforgettable Summer (1994),Drama,0.41677533221223634)
        (Farewell to Arms, A (1932),Romance War,0.41875292358311805)
        (Lady of Burlesque (1943),Comedy Mystery,0.4291252578713729)
        =====
        
        Cluster 2:
        (Angela (1995),Drama,0.26749948519341493)
        (Outlaw, The (1943),Western,0.3032690310903573)
        (Mr. Wonderful (1993),Comedy Romance,0.31810455337750054)
        (Intimate Relations (1996),Comedy,0.32754796911699535)
        (Wedding Gift, The (1994),Drama,0.34670150698634794)
        (Commandments (1997),Romance,0.3643293590048631)
        (Prefontaine (1997),Drama,0.38319693040353653)
        (Outbreak (1995),Action Drama Thriller,0.41930536322394585)
        (Target (1995),Action Drama,0.4202801350058748)
        (Mr. Jones (1993),Drama Romance,0.43168434379314075)
        (River Wild, The (1994),Action Thriller,0.44148992570133505)
        (Sword in the Stone, The (1963),Animation Children's,0.46196816522890216)
        (Apollo 13 (1995),Action Drama Thriller,0.4798544686333609)
        (Wedding Bell Blues (1996),Comedy,0.5000277075655496)
        (Tainted (1998),Comedy Thriller,0.5000277075655496)
        (Next Step, The (1995),Drama,0.5000277075655496)
        (Abyss, The (1989),Action Adventure Sci-Fi Thriller,0.5040617227719161)
        (Blue Chips (1994),Drama,0.504889914333155)
        (Touch (1997),Romance,0.5098762089560377)
        (City of Angels (1998),Romance,0.5112812907612415)
        =====
        
        Cluster 3:
        (King of the Hill (1993),Drama,0.1616109372246971)
        (All Over Me (1997),Drama,0.21163143396458606)
        (Scream of Stone (Schrei aus Stein) (1991),Drama,0.2242352672604478)
        (Land and Freedom (Tierra y libertad) (1995),War,0.26400995988074644)
        (Eighth Day, The (1996),Drama,0.26400995988074644)
        (Dadetown (1995),Documentary,0.26400995988074644)
        (Big One, The (1997),Comedy Documentary,0.26400995988074644)
        (? k?ldum klaka (Cold Fever) (1994),Comedy Drama,0.26400995988074644)
        (Girls Town (1996),Drama,0.26400995988074644)
        (Silence of the Palace, The (Saimt el Qusur) (1994),Drama,0.26400995988074644)
        (Normal Life (1996),Crime Drama,0.26400995988074644)
        (Two Friends (1986) ,Drama,0.26400995988074644)
        (Hana-bi (1997),Comedy Crime Drama,0.26400995988074644)
        (I Can't Sleep (J'ai pas sommeil) (1994),Drama Thriller,0.30648787328445026)
        (Wings of Courage (1995),Adventure Romance,0.30737955701113984)
        (All Things Fair (1996),Drama,0.3172063800749976)
        (Ed's Next Move (1996),Comedy,0.31866891253267804)
        (Sweet Nothing (1995),Drama,0.340899729487508)
        (Collectionneuse, La (1967),Drama,0.3415725162017047)
        (Love and Other Catastrophes (1996),Romance,0.3528840611405618)
        =====
        
        Cluster 4:
        (Johns (1996),Drama,0.3359609423998394)
        (Moonlight and Valentino (1995),Drama Romance,0.44546672682074695)
        (Ill Gotten Gains (1997),Drama,0.4735411201187342)
        (Stag (1997),Action Thriller,0.48845395833289823)
        (For Love or Money (1993),Comedy,0.4924464948372125)
        (Fausto (1993),Comedy,0.5318080882945031)
        (Pagemaster, The (1994),Action Adventure Animation Children's Fantasy,0.54564621
        44394557)
        (Air Up There, The (1994),Comedy,0.5465830354998226)
        (Cliffhanger (1993),Action Adventure Crime,0.6077096802914972)
        (Cops and Robbersons (1994),Comedy,0.6254443318188175)
        (House Party 3 (1994),Comedy,0.6534801085917451)
        (American Strays (1996),Action,0.6649961731283545)
        (Tokyo Fist (1995),Action Drama,0.6785878423101805)
        (Chasers (1994),Comedy,0.6898451192798156)
        (Robocop 3 (1993),Sci-Fi Thriller,0.6948210493600558)
        (Life with Mikey (1993),Comedy,0.6952268190782023)
        (It Takes Two (1995),Comedy,0.7118230658228264)
        (Shooter, The (1995),Action,0.7145247211263109)
        (Cool Runnings (1993),Comedy,0.719270923672315)
        (Window to Paris (1994),Comedy,0.7325424225594536)
        =====
     **/
          
    /**评估聚类模型 clustering mathematical evaluation**/
    //computeCost 方法，该方法通过计算所有数据点到其最近的中心点的平方和来评估聚类的效果
    //一般来说，同样的迭代次数和算法跑的次数，这个值越小代表聚类的效果越好
    // compute the cost (WCSS) on for movie and user clustering
    val movieCost = movieClusterModel.computeCost(movieVectors)
    //movieCost: Double = 2249.222165045774
    val userCost = userClusterModel.computeCost(userVectors)
    //userCost: Double = 1508.7926477117066
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
