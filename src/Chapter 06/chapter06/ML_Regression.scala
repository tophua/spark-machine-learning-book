package chapter06
import breeze.linalg.sum
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{ LinearRegressionWithSGD, LabeledPoint }
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.evaluation.RegressionMetrics
object ML_Regression {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("regression").setMaster("local[1]")
    val sc = new SparkContext(conf)
    /**每个小时自行车出租次数****/
    val records = sc.textFile("BikeSharingDataset/hour_noheader.csv").map(_.split(",")).cache()
    records.count
    //res1: Long = 17379
    records.first()
    //res0: Array[String] = Array(1, 2011-1-1, 1, 0, 1, 0, 0, 6, 0, 1, 0.24, 0.2879, 0.81, 0, 3, 13, 16)
    //将参数idx列值去重,然后对每个值使用zipWithIndex函数映射到一个唯一的索引,这样组成了一个RDD的键值映射,键是变量,值是索引
    //四季节信息(春,夏,秋,冬)
    records.map(fields => fields(2)).distinct().zipWithIndex().collectAsMap()
    //res9: scala.collection.Map[String,Long] = Map(2 -> 1, 1 -> 3, 4 -> 0, 3 -> 2)
    //将参数idx列值去重,然后对每个值使用zipWithIndex函数映射到一个唯一的索引,这样组成了一个RDD的键值映射,键是变量,值是索引
    def get_mapping(rdd: RDD[Array[String]], idx: Int): scala.collection.Map[String, Long] = {
      rdd.map(fields => fields(idx)).distinct().zipWithIndex().collectAsMap()
    }
    //对是类型变量的列(第2-9列)应用该函数
    val mappings = for (i <- Range(2, 10)) yield get_mapping(records, i)
    /**
     * mappings: scala.collection.immutable.IndexedSeq[scala.collection.Map[String,Long]] =
     * Vector
     * 春夏秋冬(Map(2 -> 1, 1 -> 3, 4 -> 0, 3 -> 2),  年份(0 2011 1 2012 )Map(1 -> 1, 0 -> 0),
     * 月份 Map(2 -> 3, 12 -> 9, 5 -> 6, 8 -> 1, 7 -> 5, 1 -> 10, 11 -> 4, 4 -> 0, 6 -> 2, 9 -> 7, 10-> 11, 3 -> 8),
     * 小时 Map(2 -> 7, 5 -> 15, 12 -> 20, 8 -> 1, 15 -> 4, 21 -> 13, 18 ->16, 7 -> 14, 1 -> 21, 17 -> 9, 23 -> 23, 4 -> 0, 11 -> 11, 14 -> 12, 20 -> 2, 6
     * -> 5, 9 -> 18, 0 -> 6, 22 -> 8, 16 -> 17, 10 -> 22, 3 -> 19, 19 -> 3, 13 -> 10),
     * 是否节假日 Map(1 -> 1, 0 -> 0),
     * 周几 Map(2 -> 3, 5 -> 4, 1 -> 6, 4 -> 0, 6 -> 1, 0 -> 2, 3 -> 5),
     * 是否工作日 Map(1 -> 1, 0 -> 0),
     * 天气类型    Map(2 -> 1, 1 -> 3, 4 -> 0, 3 -> 2)) 晴天,阴天,雪天,雨天
     */
    val cat_len = sum(mappings.map(_.size)) //合计每列
    //res8: = mappings.map(_.size)==Vector(4, 2, 12, 24, 2, 7, 2,4) sum(mappings.map(_.size)==57
    val num_len = records.first().slice(10, 14).size //取出第10列到14列 长度为4
    //res2: Array[String] = Array(0.24, 0.2879, 0.81, 0)
    val total_len = cat_len + num_len
    println("Feature vector Length for categorical features:" + cat_len) //57
    println("Feature vector Length for numerical features:" + num_len) //4
    println("Total feature vector Length:" + total_len) // 61
    //linear regression data 此部分代码最重要，主要用于产生训练数据集，按照前文所述处理类别特征和实数特征。
    //将原始数据转成二元类型特征和实数特征,并连接成长度61的特征向量
    val data = records.map { record =>
      val cat_vec = Array.ofDim[Double](cat_len) //创建61长度数组,默认都0
      var i = 0
      var step = 0 //确保整个数组特征向量位于正确的位置
      for (filed <- record.slice(2, 10)) { //获取第3列至10列的值
        val m = mappings(i) //Map 0值代表    春夏秋冬  
        val idx = m(filed) // filed列的值 获对应Map数据分类
        cat_vec(idx.toInt + step) = 1.0 //
        // println("i:"+i+"\t m:"+m+"\t filed:"+filed+"\t idx:"+idx+"\t step:"+ step+"\t idx.toInt+step:"+(idx.toInt + step)+"\t m.size:"+m.size)
        i = i + 1 //获取数据分类递增1列       
        step = step + m.size //下一列跳到数组开始位置         
      }
      val num_vec = record.slice(10, 14).map(x => x.toDouble) //获取第11列到14列值 
      val features = cat_vec ++ num_vec //数组相加
      val label = record(record.size - 1).toInt //最后列值为标签值 
      LabeledPoint(label, Vectors.dense(features))
    }
    //观察第一条记录
    val first_point = data.first()
    /**
     * first_point: org.apache.spark.mllib.regression.LabeledPoint = (16.0,[0.0,0.0,0.0
     * ,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0
     * ,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
     * ,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.24,0.2879,0.81,0.0])
     */
    println("RAW Data:" + records.first().slice(2, 20).toList)
    //res24: Array[String] = Array(1, 0, 1, 0, 0, 6, 0, 1, 0.24, 0.2879, 0.81, 0, 3, 13, 16)
    println("Lebel:" + first_point.label) //Lebel:16.0
    println("Lebel Model feature Vector:" + first_point.features)
    /**
     * Lebel Model feature Vector:[0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
     * 0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
     * 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
     * 0.0,0.0,0.0,1.0,0.24,0.2879,0.81,0.0]
     */
    println("Lebel Model feature Vector length:" + first_point.features.size) //61 

    //决策树不需要将类型数据用二元向量表示,可以直接使用原始数据 
    val data_dt = records.map { record =>
      val num_vec = record.slice(2, 14).map(x => x.toDouble) //获取第3列到14列值        
      val label = record(record.size - 1).toInt //最后列值为标签值 
      LabeledPoint(label, Vectors.dense(num_vec))
    }
    //观察第一条记录
    val first_point_dt = data_dt.first()
    println("Decision Tree feature Vector:" + first_point_dt.features)
    //Decision Tree feature Vector:[1.0,0.0,1.0,0.0,0.0,6.0,0.0,1.0,0.24,0.2879,0.81,0.0]
    println("Decision Tree feature Vector length:" + first_point_dt.features.size) //Lebel:12.0

    /***创建线性回归模型***/

    val linear_model = LinearRegressionWithSGD.train(data, 10, 0.1)
    val true_vs_predicted = data.map(p => (p.label, linear_model.predict(p.features)))
    //输出前五个真实值与预测值
    println(true_vs_predicted.take(5).toVector.toString())
    /*Vector((16.0,57.63078035304174), (40.0,55.65738194101726), (32.0,55.65738194101726),
     (13.0,55.069157795540086), (1.0,55.069157795540086))*/

    /***决策树回归模型***/
    //algo:分类Classification或者回归Regression
    //纯度 Variance
    val dt_model = DecisionTree.train(data_dt, Algo.Regression, Variance, 5)
    val preds = dt_model.predict(data_dt.map { x => x.features })
    val actual = data_dt.map { x => x.label }
    val true_vs_predicted_dt = actual.zip(preds)
    println("Decision Tree predictions:" + true_vs_predicted_dt.take(5))
    /*res37: Array[(Double, Double)] = Array((16.0,54.913223140495866), (40.0,54.913223140495866), 
    (32.0,53.171052631578945), (13.0,14.284023668639053), (1.0,14.284023668639053))*/
    println("Decision Tree depth:" + dt_model.depth) //5
    println("Decision Tree number of nodes:" + dt_model.numNodes) //63
    /**评估回归模型性能**/

    //平均方误差
    def squared_error(actual: Double, pred: Double) = {
      math.pow((actual - pred), 2)
    }
    //平均绝对误差
    def abs_error(actual: Double, pred: Double) = {
      math.abs(actual - pred)
    }
    //均方根对数误差
    def squared_log_error(actual: Double, pred: Double) = {
      //math.log返回自然对数（以e为底）的一个double值
      math.pow((math.log(pred - 1) - math.log(actual - 1)), 2)
    }
    /**计算不同度量下的性能 **/
    //线性模型度量
    //首先预测RDD实例LabledPoint中每个特征向量,然后计算预测值与实际值的误差并组成一个double数组的RDD
    //最后使用mean方法计算所有Double值的平均值
    val mse = true_vs_predicted.map { case (t, p) => squared_error(t, p) }.mean()
    //mse: Double = 45466.55394096602
    val mae = true_vs_predicted.map { case (t, p) => abs_error(t, p) }.mean()
    //mae: Double = 146.96298391737974
    val rmsle = math.sqrt(true_vs_predicted.map { case (t, p) => squared_log_error(t, p) }.mean())

    val mse2 = true_vs_predicted.map { case (v, p) => math.pow((v - p), 2) }.mean() //求平均值
    //使用内置度量工具
    val metrics = new RegressionMetrics(true_vs_predicted)
    //rmsle: Double = NaN
    println("Linear Model - Mean Squared Error: %2.4f".format(mse))
    //均方差
    println("Linear Model - Mean Squared Error: %2.4f".format(metrics.meanSquaredError))
    //Linear Model - Mean Squared Error: 45466.5539
    println("Linear Model - Mean Absolute Error: %2.4f".format(mae))
    //平均绝对误差
    println("Linear Model - Mean Absolute Error: %2.4f".format(metrics.meanAbsoluteError))
    //Linear Model - Mean Absolute Error: 146.9630
    println("Linear Model - Root Mean Squared Log Error: %2.4f".format(rmsle))
    //均方根误差
    println("Linear Model - Root Mean Squared Log Error: %2.4f".format(metrics.rootMeanSquaredError))
    //R-平方系数
    println("Linear Model - R2: %2.4f".format(metrics.r2))
    //使用内置度量工具
    val metrics_dt = new RegressionMetrics(true_vs_predicted_dt)
    //均方差
    println("Decision Tree - Mean Squared Error: %2.4f".format(metrics_dt.meanSquaredError))
    //Decision Tree - Mean Squared Error: 11560.7978
    //平均绝对误差
    println("Decision Tree - Mean Absolute Error: %2.4f".format(metrics_dt.meanAbsoluteError))
    //Decision Tree - Mean Absolute Error: 71.0969
    //均方根误差
    println("Decision Tree - Root Mean Squared Log Error: %2.4f".format(metrics_dt.rootMeanSquaredError))
    //Decision Tree - Root Mean Squared Log Error: 107.5212
    //R-平方系数
    println("Decision Tree - R2: %2.4f".format(metrics.r2))
    //Decision Tree - R2: 0.9461

    /**
     * 变换目标变量
     */
    //首先使用log函数应用到RDD LabeledPoint中的每个标签值
    val data_log = data.map(lp => LabeledPoint(math.log(lp.label), lp.features))
    //转换数据上训练线性回归模型
    val model_log = LinearRegressionWithSGD.train(data_log, 10, 0.1)
    //需要将进行指数运算计算得到预测值转换回原始值使用exp函数
    val true_vs_predicted_log = data_log.map(p => (math.exp(p.label), math.exp(model_log.predict(p.features))))
    //使用内置度量工具
    val metrics_log = new RegressionMetrics(true_vs_predicted_log)

    //均方差
    println("Mean Squared Error: %2.4f".format(metrics_log.meanSquaredError))
    //Mean Squared Error: 66501.5660   
    //平均绝对误差
    println("Mean Absolute Error: %2.4f".format(metrics_log.meanAbsoluteError))
    //Mean Absolute Error: 183.9396
    //均方根误差
    println("Root Mean Squared Log Error: %2.4f".format(metrics_log.rootMeanSquaredError))
    //Root Mean Squared Log Error: 257.8790

    println("Non log-transformed predictions:\n" + true_vs_predicted.take(3))
    //res61:Array((16.0,57.63078035304174), (40.0,55.65738194101726), (32.0,55.65738194101726))
    println("Log-transformed predictions:\n" + true_vs_predicted_log.take(3))
    //res62:Array((15.999999999999998,4.063614487622041), (40.0,3.88642556430256),(32.0,3.88642556430256))
    /**
     * 总结:结果和原始数据训练模型性能 比较,可以看到我们提升了RMSLE(平均绝对误差)的性能,但是却没有提升MSE和MAE的性能
     */
    //决策树模型性能测试
    val data_dt_log = data_dt.map(lp => LabeledPoint(math.log(lp.label), lp.features))
    val dt_model_log = DecisionTree.train(data_dt_log, Algo.Regression, Variance, 5)
    val preds_log = dt_model_log.predict(data_dt_log.map(p => p.features))
    val actual_log = data_dt_log.map(p => p.label)
    val true_vs_predicted_dt_log = actual_log.zip(preds_log).map { case (t, p) => (math.exp(t), math.exp(p)) }
    val metrics_dt_log = new RegressionMetrics(true_vs_predicted_dt_log)
    //均方差
    println("Mean Squared Error: %2.4f".format(metrics_dt_log.meanSquaredError))
    //Mean Squared Error: 14781.5760
    //平均绝对误差
    println("Mean Absolute Error: %2.4f".format(metrics_dt_log.meanAbsoluteError))
    //Mean Absolute Error: 76.4131
    //均方根误差
    println("Root Mean Squared Log Error: %2.4f".format(metrics_dt_log.rootMeanSquaredError))
    //Root Mean Squared Log Error: 121.5795
    println("Non log-transformed predictions:\n" + true_vs_predicted_dt.take(3))
    //res69:Array((16.0,54.913223140495866), (40.0,54.913223140495866), (32.0,53.171052631578945))
    println("Log-transformed predictions:\n" + true_vs_predicted_dt_log.take(3))
    //res70:Array((15.999999999999998,37.53077978715452), (40.0,37.53077978715452), (32.0,7.279707099390729))
    /**
     * 总结:表明决策树在变换后的性能有所下降
     */
     val num_data = records.count()
    //模型参数调优
     //暂时无用
    val data_with_idx = data.zipWithIndex().map{case(k, v)=> (v, k)}
    
  
    val trainTestSplit = data.randomSplit(Array(0.8, 0.2), 42)
    
    //训练数据集
    val train_data = trainTestSplit(0)
    //测试数据集
    val test_data = trainTestSplit(1)
    println("Training data size: %d".format(train_data.count))
    println("Test data size: %d".format(test_data.count))
    println("Total data size: %d ".format( num_data))
    println("Train + Test size : %d".format(train_data.count + test_data.count))
    //使用同样的方法提取决策树模型所需特征
    
    val data_with_idx_dt = data_dt.zipWithIndex().map{case(k, v)=> (v, k)}
    val trainTestSplit_dt = data_with_idx_dt.randomSplit(Array(0.8, 0.2), 42)
    //训练数据集
    val train_data_dt = trainTestSplit_dt(0)
    //测试数据集
    val test_data_dt = trainTestSplit_dt(1)
    println("Training data size: %d".format(train_data_dt.count))
    println("Test data size: %d".format(test_data_dt.count))
    println("Total data size: %d ".format( num_data))
    println("Train + Test size : %d".format(train_data_dt.count + test_data_dt.count))
    
   def trainWithParams(train: RDD[LabeledPoint],test: RDD[LabeledPoint], regParam: Double,numIterations: Int,  stepSize: Double) = {
      val lr = new LinearRegressionWithSGD
      //numIterations迭代次数,stepSize步长,regParam
      lr.optimizer.setNumIterations(numIterations).setRegParam(regParam).setStepSize(stepSize)
      val model=lr.run(train)
      val tp=test.map { x => (x.label,model.predict(x.features))}
      val rmsle = math.sqrt(tp.map{case(t, p)=> squared_log_error(t, p)}.mean())
      // val metrics_dt_log = new RegressionMetrics(tp)
       //使用均方根对数评估指标
      // println(">>>>>>>>>>>"+metrics_dt_log.rootMeanSquaredError)
       //metrics_dt_log.rootMeanSquaredError
      println(">>>>>>>>>>>"+rmsle)
       rmsle
       
    }
    // num iterations
    /**迭代次数,(性能测试)**/
    val iterResults = Seq(1, 5, 10, 50,10,100).map { param =>
       trainWithParams(train_data,test_data, 0.0, param,1.0)
    }
     println(iterResults)

  }
  
  //将参数idx列值去重,然后对每个值使用zipWithIndex函数映射到一个唯一的索引,这样组成了一个RDD的键值映射,键是变量,值是索引
  def get_mapping(rdd: RDD[Array[String]], idx: Int) =
    {
      rdd.map(filed => filed(idx)).distinct().zipWithIndex().collectAsMap()
    }
}
