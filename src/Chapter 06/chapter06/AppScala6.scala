package chapter06

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
 * 构建回归模型----连续型的数据
 */
object AppScala6 {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    /**每个小时自行车出租次数****/
    val rawData = sc.textFile("BikeSharingDataset/hour_noheader.csv")
    val records = rawData.map(line => line.split(","))
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
    
    val mappings=for( i<-2 to 10){
      get_mapping(records,i)
    }


    /***从数据中抽取合适的特征**/
    // val records = rawData.map(line => line.split("\t"))
    //开始四列分别包含URL，页面的ID，原始的文本内容和分配给页面的类别
    // records.first
    //Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...
    //由于数据格式的问题，我们做一些数据清理的工作：把额外的(“)去掉，。   
    val data = records.map { r =>
      //把额外的(“)去掉
      val trimmed = r.map(_.replaceAll("\"", ""))
      /*  val trimmed = r.map { x =>
        //println("befor:" + x)
        val v = x.replaceAll("\"", "")
        // println("after:" + v)
        v
      }*/

      //println("r:" + r.toList + "\t size:" + r.size)
      //r是Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...
      //数据一条数据为0开始      
      //println(r.size - 1 + ":\t第一条:" + r(0) + ":\t最后一个数据size:" + r(r.size - 1))
      /* r.foreach {
        var i = 0
        x =>
          println(i + ">>>" + x)
          i += 1
      }*/
      //r.size - 1把所有列"\"",替换成 ""    
      //取出最后一列值转换成toInt,一般0和1
      val label = trimmed(r.size - 1).toInt
      println("(最后一列值)label:" + label)
      //创建一个迭代器返回由这个迭代器所产生的值区间,取子集set(1,4为元素位置, 从0开始),从位置4开始,到数组的长度     
      //slice提取第5列开始到25列的特征矩阵
      //数据集中缺失数据为?,直接用0替换缺失数据
      val features = trimmed.slice(4, r.size - 1).map {
        d =>
          //println("" + d)
          if (d == "?") {
            //println(d)
            0.0
          } else {
            d.toDouble
          }
      }
      //将标签和特征向量转换为LabeledPoint为实例,将特征向量存储到MLib的Vectors中,label一般是0和1
      LabeledPoint(label, Vectors.dense(features))
    }
    data.cache
    // numData: Long = 7395
    val numData = data.count
    // train a Logistic Regression model    
    // note that some of our data contains negative feature vaues. For naive Bayes we convert these to zeros
    //创建朴素贝叶斯模型,要求特征值非负,否则碰到负的特征值程序会抛出错误
    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      //如果最后一列小于0,则为0
      val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }

    /***创建训练分类模型**/
    //设置逻辑回归和SVM迭代次数
    val numIterations = 10
    //决策树最大深度
    val maxTreeDepth = 5
    //创建逻辑回归模型
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
    //创建训练SVM模型
    val svmModel = SVMWithSGD.train(data, numIterations)
    //创建朴素贝叶斯模型,使用没有负特征的数据
    // note we use nbData here for the NaiveBayes model training
    val nbModel = NaiveBayes.train(nbData)
    //创建决策树
    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

    /***使用分类模型预测**/
    // make prediction on a single data point    
    val dataPoint = data.first
    //逻辑模型预测
    // dataPoint: org.apache.spark.mllib.regression.LabeledPoint = LabeledPoint(0.0, [0.789131,2.055555556,0.676470588, ...
    //训练数据中第一个样本,模型预测值为1,即长久
    val prediction = lrModel.predict(dataPoint.features)
    //模型预测出错了
    // prediction: Double = 1.0
    //检验一下这个样本真正的标签 
    val trueLabel = dataPoint.label
    // trueLabel: Double = 0.0
    //将RDD[Vector]整体作为输入做预测
    val predictions = lrModel.predict(data.map(lp => lp.features))
    predictions.take(5)
    // res1: Array[Double] = Array(1.0, 1.0, 1.0, 1.0, 1.0)
    //SVM模型
    val predictionsSvmModel = svmModel.predict(data.map(lp => lp.features))
    predictionsSvmModel.take(5)
    //NaiveBayes模型朴素贝叶斯
    val predictionsNbModel = nbModel.predict(data.map(lp => lp.features))
    predictionsNbModel.take(5)
    ///创建决策树
    val predictionsDtModel = dtModel.predict(data.map(lp => lp.features))
    predictionsDtModel.take(5)

    /***评估分类模型的性能**/
    /**
     * 二分类中使用的评估方法包括
     * 1)预测正确率和错误率
     * 2)准确率和召回率
     * 3)准确率一召回率曲线下方的面积ROC曲线
     * 4)准确率和召回率
     *
     */
    //预测的正确率和错误率
    //正确率=训练样本中被正确分类的数目/总样本数
    //错误率=训练样本中被错误分类的样本数目/总样本数

    //逻辑回归模型对输入特征预测值与实际标签进行比较求和
    val lrTotalCorrect = data.map { point =>
      //point.label实际标签进行比较
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    // lrTotalCorrect: Double = 3806.0
    /**正确率 将对正确分类的样本数目求和并除以样本总数,逻辑回归模型得到正确率51.5%**/
    val lrAccuracy = lrTotalCorrect / numData
    // lrAccuracy: Double = 0.5146720757268425       

    //SVM模型对输入特征预测值与实际标签进行比较求和
    val svmTotalCorrect = data.map { point =>
      //point.label实际标签进行比较
      if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    //朴素贝叶斯模型对输入特征预测值与实际标签进行比较求和
    val nbTotalCorrect = nbData.map { point =>
      if (nbModel.predict(point.features) == point.label) 1 else 0
    }.sum
    //决策树模型对输入特征预测值与实际标签进行比较求和
    // decision tree threshold needs to be specified
    val dtTotalCorrect = data.map { point =>
      val score = dtModel.predict(point.features)
      //决策树的预测阈值需要明确给出 0.5
      val predicted = if (score > 0.5) 1 else 0
      if (predicted == point.label) 1 else 0
    }.sum
    //预测SVM模型正确率,逻辑回归模型得到正确率51.4%/
    val svmAccuracy = svmTotalCorrect / numData
    //svmAccuracy: Double = 0.5146720757268425
    //预测朴素贝叶斯模型正确率58%
    val nbAccuracy = nbTotalCorrect / numData
    // nbAccuracy: Double = 0.5803921568627451
    //决策树斯模型正确率 65%
    val dtAccuracy = dtTotalCorrect / numData
    // dtAccuracy: Double = 0.6482758620689655
    /***结论 SVM和朴素贝叶斯模型性能都较差,而决策树模型正确率65%,但还不是很高***/

    /**
     * *
     * 准确率和召回率
     * 准确率通常用于评价结果和质量,召回率用来评价结果的完整性
     * 二分类准确率=真阳性的数目除以真阳性和假阳性的总数,
     * 真阳性是指被正确预测的类别为1的样本
     * 假阳性是指被错误预测的类别为1的样本
     * *
     */
    //计算二分类的PR(召回率)和ROC曲线下的面积
    val metrics = Seq(lrModel, svmModel).map { model =>
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
    // again, we need to use the special nbData for the naive Bayes metrics 
    val nbMetrics = Seq(nbModel).map { model =>
      val scoreAndLabels = nbData.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
    // here we need to compute for decision tree separately since it does 
    // not implement the ClassificationModel interface
    val dtMetrics = Seq(dtModel).map { model =>
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
    }
    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
    allMetrics.foreach {
      case (m, pr, roc) =>
        println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
    }
    /*
          平均准确率,得到模型的平均率都差不多
    LogisticRegressionModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
    SVMModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
    NaiveBayesModel, Area under PR: 68.0851%, Area under ROC: 58.3559%
    DecisionTreeModel, Area under PR: 74.3081%, Area under ROC: 64.8837%
    */

    /***特征数据标准化****/
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    /**将特征向量用RowMatrix类表示成MLib中的分布式矩阵,RowMatrix是一个由向量组成的RDD,其中每个向量是分布矩阵的一行**/
    val vectors = data.map {
      lp =>
        //每列的特征
        println("lp.features:" + lp.features)
        lp.features
    }
    val matrix = new RowMatrix(vectors)
    //计算矩阵每列的统计特性 
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //矩阵每列的平均值
    println(matrixSummary.mean)
    // [0.41225805299526636,2.761823191986623,0.46823047328614004, ...
    //矩阵每列的最小值
    println(matrixSummary.min)
    // [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.045564223,-1.0, ...
    //矩阵每列的最大值
    println(matrixSummary.max)
    // [0.999426,363.0,1.0,1.0,0.980392157,0.980392157,21.0,0.25,0.0,0.444444444, ...
    //矩阵每列的方差,发现第一个方差和均值都比较高,不符合标准的高斯分布
    println(matrixSummary.variance)
    // [0.1097424416755897,74.30082476809638,0.04126316989120246, ...
    //矩阵每列中的非0荐的数目
    println(matrixSummary.numNonzeros)
    //对每个特征进行标准化,使得每个特征是0均值和单位标准差,具体做法是对每个特征值减去列的均值,然后除以列的标准差
    //进行缩放
    import org.apache.spark.mllib.feature.StandardScaler
    //第一个参数是否从数据中减去均值,另一个表示是否应用标准差缩放,返回归一化的向量
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    //使用Map保留LabeledPoint数据集的标签,transform数据标准化
    val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
    // compare the raw features with the scaled features
    //标准化之前数据
    println(data.first.features)
    // [0.789131,2.055555556,0.676470588,0.205882353,
    //标准化之后的数据
    println(scaledData.first.features)
    // [1.1376439023494747,-0.08193556218743517,1.025134766284205,-0.0558631837375738,
    //可以看出第一个特征已经应用标准差公式被转换,验证第一个特征值,第一特征值减去均值,然后除以标准差(方差的平方根)
    println((0.789131 - 0.41225805299526636) / math.sqrt(0.1097424416755897))
    // 1.137647336497682
    /***标准化数据重新训练模型,这里只训练逻辑回归(决策树和朴素贝叶斯不受特征标准话的影响),并说明特征标准化的影响***/
    //创建逻辑回归模型
    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    //逻辑回归模型对输入特征预测值与实际标签进行比较求和
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    /**正确率 将对正确分类的样本数目求和并除以样本总数,逻辑回归模型得到正确率62%**/
    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    // lrAccuracyScaled: Double = 0.6204192021636241

    val lrPredictionsVsTrue = scaledData.map { point =>
      //
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    //准确率和召回率
    val lrPr = lrMetricsScaled.areaUnderPR
    //
    val lrRoc = lrMetricsScaled.areaUnderROC
    println(f"${lrModelScaled.getClass.getSimpleName}\n 正确率Accuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")

    /*
           从结果可以看出,通过简单的特征标准化,就提高了逻辑回归的准确率,并将AUC从随机50%,提升到62%
    LogisticRegressionModel
    Accuracy: 62.0419%
    Area under PR: 72.7254%
    Area under ROC: 61.9663%
    */

    /**其他数据特征对模型影响**/

    // Investigate the impact of adding in the 'category' feature
    //每个类别做一个去重索引映射,这里索引可以用于类别特征做1-of-K编码
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    // categories: scala.collection.immutable.Map[String,Int] = Map("weather" -> 0, "sports" -> 6, 
    //	"unknown" -> 4, "computer_internet" -> 12, "?" -> 11, "culture_politics" -> 3, "religion" -> 8,
    // "recreation" -> 2, "arts_entertainment" -> 9, "health" -> 5, "law_crime" -> 10, "gaming" -> 13, 
    // "business" -> 1, "science_technology" -> 7)

    val numCategories = categories.size
    // numCategories: Int = 14
    //我们需要创建一个长为14向量来表示类别特征,然后根据每个样本所属类别索引,对应相应的维度赋值为1,其他为0
    val dataCategories = records.map { r =>
      //把额外的(“)去掉
      val trimmed = r.map(_.replaceAll("\"", ""))
      //取出最后一列值转换成toInt,一般0和1
      val label = trimmed(r.size - 1).toInt
      //根据每个样本所属类别索引,对应相应的维度赋值为1,其他为0
      val categoryIdx = categories(r(3))
      //ofDim创建几行几列二维数组
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      //创建一个迭代器返回由这个迭代器所产生的值区间,取子集set(1,4为元素位置, 从0开始),从位置4开始,到数组的长度     
      //slice提取第5列开始到25列的特征矩阵
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
    println(dataCategories.first)
    // LabeledPoint(0.0, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,
    //	0.676470588,0.205882353,0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,
    // 0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])
    // 使用StandardScaler方法进行标准化转换
    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
    println(dataCategories.first.features)
    // [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,
    // 0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,
    // 5424.0,170.0,8.0,0.152941176,0.079129575]
    println(scaledDataCats.first.features)
    /*
    [-0.023261105535492967,2.720728254208072,-0.4464200056407091,-0.2205258360869135,-0.028492999745483565,
    -0.2709979963915644,-0.23272692307249684,-0.20165301179556835,-0.09914890962355712,-0.381812077600508,
    -0.06487656833429316,-0.6807513271391559,-0.2041811690290381,-0.10189368073492189,1.1376439023494747,
    -0.08193556218743517,1.0251347662842047,-0.0558631837375738,-0.4688883677664047,-0.35430044806743044
    ,-0.3175351615705111,0.3384496941616097,0.0,0.8288021759842215,-0.14726792180045598,0.22963544844991393,
    -0.14162589530918376,0.7902364255801262,0.7171932152231301,-0.29799680188379124,-0.20346153667348232,
    -0.03296720969318916,-0.0487811294839849,0.9400696843533806,-0.10869789547344721,-0.2788172632659348]
    */
    /***评估性能 **/
    // train model on scaled data and evaluate metrics
    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    //准确率
    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")
    /*
     * 总结对数据标准化,模型准确率得到提升,将50%提升到62%,之后类别特征,模型性能进一步提升到65%(其中新添加的特征也做了标准化操作)
    LogisticRegressionModel
    Accuracy: 66.5720%
    Area under PR: 75.7964%
    Area under ROC: 66.5483%
    */
    /**重新训练朴素贝叶斯模型**/
    // train naive Bayes model with only categorical data
    val dataNB = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      LabeledPoint(label, Vectors.dense(categoryFeatures))
    }
    val nbModelCats = NaiveBayes.train(dataNB)
    /**重新训练朴素贝叶斯模型对性能评估**/
    val nbTotalCorrectCats = dataNB.map { point =>
      if (nbModelCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracyCats = nbTotalCorrectCats / numData
    val nbPredictionsVsTrueCats = dataNB.map { point =>
      (nbModelCats.predict(point.features), point.label)
    }
    val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
    val nbPrCats = nbMetricsCats.areaUnderPR
    val nbRocCats = nbMetricsCats.areaUnderROC
    println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")
    /*
     * 使用格式正确的输入数据后,朴素贝叶斯的性能从58%提升到60% 
    NaiveBayesModel
    Accuracy: 60.9601%
    Area under PR: 74.0522%
    Area under ROC: 60.5138%
    */
    /**模型参数调优**/
    import org.apache.spark.rdd.RDD
    import org.apache.spark.mllib.optimization.Updater
    import org.apache.spark.mllib.optimization.SimpleUpdater
    import org.apache.spark.mllib.optimization.L1Updater
    import org.apache.spark.mllib.optimization.SquaredL2Updater
    import org.apache.spark.mllib.classification.ClassificationModel
    /**线性模型参数**/
    // helper function to train a logistic regresson model
    //给定输入训练模型
    def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
      val lr = new LogisticRegressionWithSGD
      lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
      lr.run(input)
    }
    // helper function to create AUC metric
    //定义第二个辅助函数并根据输入数据和分类模型,计算相关的AUC
    def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
      val scoreAndLabels = data.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (label, metrics.areaUnderROC)
    }
    //缓存标准化数据,加快多次模型训练的速度
    // cache the data to increase speed of multiple runs agains the dataset
    scaledDataCats.cache
    // num iterations
    /**迭代次数,(性能测试)**/
    val iterResults = Seq(1, 5, 10, 50).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
      createMetrics(s"$param iterations", scaledDataCats, model)
    }
    iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     * 一旦完成特定次数的迭代,再增大迭代次数对结果影响较小
    1 iterations, AUC = 64.97%
    5 iterations, AUC = 66.62%
    10 iterations, AUC = 66.55%
    50 iterations, AUC = 66.81%
    */
    /**步长次数,(性能测试),步长用来控制算法最陡的梯度方向上应该前进多远**/
    // step size
    val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
      createMetrics(s"$param step size", scaledDataCats, model)
    }
    stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     * 步长增长过大对性能有负面影响
    0.001 step size, AUC = 64.95%
    0.01 step size, AUC = 65.00%
    0.1 step size, AUC = 65.52%
    1.0 step size, AUC = 66.55%
    10.0 step size, AUC = 61.92%
    */
    /**正则化通过限制模型的复杂度避免模型在训练数据中过拟合regularization**/
    val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
    }
    regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     *结论:等级低的正化对模型的性能影响不大,然后,增大正则化可以看到欠拟合会导致较低模型性能 
    0.001 L2 regularization parameter, AUC = 66.55%
    0.01 L2 regularization parameter, AUC = 66.55%
    0.1 L2 regularization parameter, AUC = 66.63%
    1.0 L2 regularization parameter, AUC = 66.04%
    10.0 L2 regularization parameter, AUC = 35.33%
    */

    /**决策树模型参数**/
    /**决策树模型在一开始使用原始数据做训练时获得了最好的性能,当时设置了参数最大深度**/
    import org.apache.spark.mllib.tree.impurity.Entropy
    import org.apache.spark.mllib.tree.impurity.Gini
    def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
      DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
    }

    // investigate tree depth impact for Entropy impurity
    val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainDTWithParams(data, param, Entropy)
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.areaUnderROC)
    }
    dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     * 
      1 tree depth, AUC = 59.33%
      2 tree depth, AUC = 61.68%
      3 tree depth, AUC = 62.61%
      4 tree depth, AUC = 63.63%
      5 tree depth, AUC = 64.88%
      10 tree depth, AUC = 76.26%
      20 tree depth, AUC = 98.45%
      */
    // investigate tree depth impact for Gini impurity
    val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
      val model = trainDTWithParams(data, param, Gini)
      val scoreAndLabels = data.map { point =>
        val score = model.predict(point.features)
        (if (score > 0.5) 1.0 else 0.0, point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param tree depth", metrics.areaUnderROC)
    }
    dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     * 结论:提高树的深度可以得到更精确的模型,然而树的深度越大,模型对训练数据过拟合度越严重
     * 两种不纯度方法对性能的影响差异较小 
    1 tree depth, AUC = 59.33%
    2 tree depth, AUC = 61.68%
    3 tree depth, AUC = 62.61%
    4 tree depth, AUC = 63.63%
    5 tree depth, AUC = 64.89%
    10 tree depth, AUC = 78.37%
    20 tree depth, AUC = 98.87%
    */
    /**朴素贝叶斯模型参数**/
    // investigate Naive Bayes parameters
    def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
      val nb = new NaiveBayes
      nb.setLambda(lambda)
      nb.run(input)
    }
    val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
      val model = trainNBWithParams(dataNB, param)
      val scoreAndLabels = dataNB.map { point =>
        (model.predict(point.features), point.label)
      }
      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
      (s"$param lambda", metrics.areaUnderROC)
    }
    nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
    /*
     * Lamda参数对朴素贝叶斯模型的影响,Lamda解决数据中某个类别和某个特征值的组合没有同时出现的问题 
     * 总结:Lamda的值对性能没有影响
    0.001 lambda, AUC = 60.51%
    0.01 lambda, AUC = 60.51%
    0.1 lambda, AUC = 60.51%
    1.0 lambda, AUC = 60.51%
    10.0 lambda, AUC = 60.51%
    */
    /**交叉验证参数**/
    // illustrate cross-validation
    // create a 60% / 40% train/test data split
    val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
    val train = trainTestSplit(0)
    val test = trainTestSplit(1)
    // now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
    // in addition, we will evaluate the differing performance of regularization on training and test datasets
    val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", test, model)
    }
    regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
    /*
    0.0 L2 regularization parameter, AUC = 66.480874%
    0.001 L2 regularization parameter, AUC = 66.480874%
    0.0025 L2 regularization parameter, AUC = 66.515027%
    0.005 L2 regularization parameter, AUC = 66.515027%
    0.01 L2 regularization parameter, AUC = 66.549180%
    */

    // training set results
    val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
      val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
      createMetrics(s"$param L2 regularization parameter", train, model)
    }
    regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
    /*
    0.0 L2 regularization parameter, AUC = 66.260311%
    0.001 L2 regularization parameter, AUC = 66.260311%
    0.0025 L2 regularization parameter, AUC = 66.260311%
    0.005 L2 regularization parameter, AUC = 66.238294%
    0.01 L2 regularization parameter, AUC = 66.238294%
    */
  }
}