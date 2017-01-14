package Chapter01

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object AppScala {
  def main(args: Array[String]) {
   //val sparkConf = new SparkConf().setMast("local[2]").setAppName("SparkHdfsLR")
    
    
    val conf = new SparkConf().setAppName("test").setMaster("local") 
   /** val conf = new SparkConf().setMaster("spark://dept3:8088").setAppName("Chapter01")
       .set("spark.driver.port", "8088")
      .set("spark.fileserver.port", "3306")
      .set("spark.replClassServer.port", "8080")
      .set("spark.broadcast.port", "8089")
      .set("spark.blockManager.port", "15000")**/
    //val  sc = new SparkContext(conf)
   val sc = new SparkContext("local[2]", "First Spark App")
    // we take the raw data in CSV format and convert it into a set of records of the form (user, product, price)
    //将csv格式的原始数据转化为(user, product, price),返回三元数组(客户名称,商品名称,商品价格)
    
    /**
     * 数据格式:
     * 客户		   商品名称							价格
     * John	   iPhone Cover	        9.99
		 * John	   Headphones	          5.49
		 * Jack	   iPhone Cover	        9.99
		 * Jill	   Samsung Galaxy Cover	8.95
		 * Bob	   iPad Cover	          5.49
     */
      //hdfs://xcsq:8089/machineLearning/Chapter01/UserPurchaseHistory.csv
    //不能指定文件名
    //val data = sc.textFile("hdfs://xcsq:8089/machineLearning/Chapter01/")
    val data = sc.textFile("D:\\eclipse44_64\\workspace\\MachineLearning\\data\\UserPurchaseHistory.csv")
      .map(line => line.split(","))
      .map(purchaseRecord => (purchaseRecord(0), purchaseRecord(1), purchaseRecord(2)))
    // let's count the number of purchases
    //购买数据总数
    val numPurchases = data.count()
    // let's count how many unique users made purchases
    //有多少个不同客户购买过商品
    val uniqueUsers = data.map { case (user, product, price) => {
      println(user)
      user     
     }
    }.distinct().count()
    // let's sum up our total revenue
    //得出总收入
    val totalRevenue = data.map { case (user, product, price) => price.toDouble }.sum()
    // let's find our most popular product
    //求最畅销的产品是什么,
    val productsByPopularity = data
      .map { case (user, product, price) => (product, 1) }.sortByKey(false)
      .reduceByKey(_ + _)  // .sortByKey(false)(降序)排序 同时可以使用sortByKey()来对其进行并行操作排序
      .collect()
     // .sortBy(-_._2)//-表示降序 按着购买次数进行(降序)排序,本地排序
    val mostPopular = productsByPopularity(0)//
    // finally, print everything out
    //5次交易信息
    println("Total purchases: " + numPurchases)
     //总有客户
    println("Unique users: " + uniqueUsers)
    //总收入
    println("Total revenue: " + totalRevenue)
    //最畅销的产品,共购买次数
    println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))

    sc.stop()
  }
}