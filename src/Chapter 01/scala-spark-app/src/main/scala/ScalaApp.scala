import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.doubleRDDToDoubleRDDFunctions
import org.apache.spark.SparkContext.rddToPairRDDFunctions

/**
 * A simple Spark app in Scala
 */
object ScalaApp {

  def main(args: Array[String]) {
   //val sparkConf = new SparkConf().setMast("local[2]").setAppName("SparkHdfsLR")
    
    
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val  sc = new SparkContext(conf)
   // val sc = new SparkContext("local[2]", "First Spark App")

    // we take the raw data in CSV format and convert it into a set of records of the form (user, product, price)
    //将csv格式的原始数据转化为(user, product, price),客户名称,商品名称,商品价格
    val data = sc.textFile("data/UserPurchaseHistory.csv")
      .map(line => line.split(","))
      .map(purchaseRecord => (purchaseRecord(0), purchaseRecord(1), purchaseRecord(2)))

    // let's count the number of purchases
      //购买次数
    val numPurchases = data.count()

    // let's count how many unique users made purchases
    //求有多少个不同客户购买过商品
    val uniqueUsers = data.map { case (user, product, price) => user }.distinct().count()

    // let's sum up our total revenue
    //求和得出总收入
    val totalRevenue = data.map { case (user, product, price) => price.toDouble }.sum()

    // let's find our most popular product
    //求最畅销的产品是什么
    val productsByPopularity = data
      .map { case (user, product, price) => (product, 1) }
      .reduceByKey(_ + _)
      .collect()
      .sortBy(-_._2)//-表示降序 按着购买次数进行(降序)排序
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
