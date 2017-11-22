import java.io.PrintWriter
import java.net.ServerSocket
import java.text.{SimpleDateFormat, DateFormat}
import java.util.Date

import org.apache.spark.SparkContext._
import org.apache.spark.streaming.{Seconds, StreamingContext}

import scala.util.Random

/**
 * A producer application that generates random "product events", up to 5 per second, and sends them over a
 * network connection
 */
object StreamingProducer {

  def main(args: Array[String]) {

    val random = new Random()

    // Maximum number of events per second
    //每秒事件的最大数量
    val MaxEvents = 6

    // Read the list of possible names
    //读取可能名称的列表,namesResource: InputStream
    val namesResource = this.getClass.getResourceAsStream("/names.csv")
    //names: Seq[String]
    val names = scala.io.Source.fromInputStream(namesResource)
      .getLines()//行读取
      .toList//转换列表
      .head//取出第一行
      .split(",")//分隔逗号数据
      .toSeq

    // Generate a sequence of possible products
    //生成可能的产品序列
    val products = Seq(
      "iPhone Cover" -> 9.99,
      "Headphones" -> 5.49,
      "Samsung Galaxy Cover" -> 8.95,
      "iPad Cover" -> 7.49
    )

    /** 
     *  Generate a number of random product events 
     *  产生多项随机产品事件
     *  */
    def generateProductEvents(n: Int) = {
      (1 to n).map { i =>
        //随机产生一个产品和价格
        val (product, price) = products(random.nextInt(products.size))
        //根据Seq随机产生shuffle
        val user = random.shuffle(names).head
        (user, product, price)
      }
    }

    // create a network producer
    //创建一个网络服务,监听端口9999
    val listener = new ServerSocket(9999)
    println("Listening on port: 9999")

    while (true) {
      //接受监听
      val socket = listener.accept()
      //启动线程
      new Thread() {
        override def run = {
          println("Got client connected from: " + socket.getInetAddress)
          //
          val out = new PrintWriter(socket.getOutputStream(), true)

          while (true) {
            Thread.sleep(1000)
            //随机最大种子秒
            val num = random.nextInt(MaxEvents)            
            val productEvents = generateProductEvents(num)
            productEvents.foreach{ event =>
              out.write(event.productIterator.mkString(","))
              out.write("\n")
            }
            out.flush()
            println(s"Created $num events...")
          }
          socket.close()
        }
      }.start()
    }
  }
}

/**
 * A simple Spark Streaming app in Scala
 * 简单的Spark Streaming程序
 */
object SimpleStreamingApp {

  def main(args: Array[String]) {
   //
    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    // here we simply print out the first few elements of each batch
    //在这里,我们只需打印出每个批次的前几个元素
    println("==================")
    stream.print()
    ssc.start()
    ssc.awaitTermination()

  }
}

/**
 * A more complex Streaming app, which computes statistics and prints the results for each batch in a DStream
 * 批量打印
 */
object StreamingAnalyticsApp {

  def main(args: Array[String]) {

    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    val stream = ssc.socketTextStream("localhost", 9999)

    // create stream of events from raw text elements
    //从原始文本元素创建事件流
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2))
    }

    /*
      We compute and print out stats for each batch.
      Since each batch is an RDD, we call forEeachRDD on the DStream, and apply the usual RDD functions
      we used in Chapter 1.
      foreachRDD来把Dstream中的数据发送到外部的文件系统中,外部文件系统主要是数据库
     */
    events.foreachRDD { (rdd, time) =>
      val numPurchases = rdd.count()//总数
      val uniqueUsers = rdd.map { case (user, _, _) => user }.distinct().count()
      val totalRevenue = rdd.map { case (_, _, price) => price.toDouble }.sum()//合计
      val productsByPopularity = rdd
        .map { case (user, product, price) => (product, 1) }//产品合计
        .reduceByKey(_ + _)
        .collect()
        .sortBy(-_._2)//倒序
      val mostPopular = productsByPopularity(0)

      val formatter = new SimpleDateFormat
      //批量收到数据时间
      val dateStr = formatter.format(new Date(time.milliseconds))
      println(s"== Batch start time: $dateStr ==")
      println("Total purchases: " + numPurchases)//购买总数
      println("Unique users: " + uniqueUsers)//购买用户数
      println("Total revenue: " + totalRevenue)//收入
      //购买多用户和产品
      println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    }

    // start the context
    ssc.start()
    ssc.awaitTermination()

  }

}

object StreamingStateApp {
  import org.apache.spark.streaming.StreamingContext._
  /**
   * 状态更新 
   */
  def updateState(prices: Seq[(String, Double)], currentTotal: Option[(Int, Double)]) = {
    //通过Spark内部的reduceByKey按key规约，然后这里传入某key当前批次的Seq/List,再计算当前批次的总和
    val currentRevenue = prices.map(_._2).sum
 
    val currentNumberPurchases = prices.size
    //已累加的值
    val state = currentTotal.getOrElse((0, 0.0))
    Some((currentNumberPurchases + state._1, currentRevenue + state._2))
  }

  def main(args: Array[String]) {

    val ssc = new StreamingContext("local[2]", "First Streaming App", Seconds(10))
    // for stateful operations, we need to set a checkpoint location
    //使用updateStateByKey前需要设置checkpoint
    ssc.checkpoint("/tmp/sparkstreaming/")
    val stream = ssc.socketTextStream("localhost", 9999)

    // create stream of events from raw text elements
    //从原始文本元素创建事件流
    val events = stream.map { record =>
      val event = record.split(",")
      (event(0), event(1), event(2).toDouble)
    }

    val users = events.map{ case (user, product, price) => (user, (product, price)) }
    //updateStateByKey在有新的数据信息进入或更新时，可以让用户保持想要的任何状
    val revenuePerUser = users.updateStateByKey(updateState)
    revenuePerUser.print()

    // start the context
    ssc.start()
    ssc.awaitTermination()

  }
}
