package chapater03

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.doubleRDDToDoubleRDDFunctions
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.util.StatCounter

/**
 * A simple Spark app in Scala
 */
object ScalaApp3 {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    val user_data = sc.textFile("ml-100k/u.user")
    val user_first = user_data.first()
    //用户ID|年龄  |性别|   职业         | 邮编
    //1     |24  |  M |technician| 85711
    //统计用户,性别,职业和邮编的数目
    val usre_files = user_data.map(line => line.split('|')) //以竖线分隔
    println(usre_files)
    //统计用户数
    val num_users = usre_files.map(line => line.head).count()
    //统计性别的数目
    val num_genders = usre_files.map { line => line(2) }.distinct().count()
    //统计职业的数目
    val num_occuptions = usre_files.map { line => line(3) }.distinct().count()
    //统计邮编的数目
    val num_zipCode = usre_files.map { line => line(4) }.distinct().count()
    //println("Most popular product: %s with %d purchases".format(mostPopular._1, mostPopular._2))
    println("Users: %d,genders: %d,occupations: %d,ZIP Codes: %d".format(num_users, num_genders, num_occuptions, num_zipCode))
    //电影数据
    val move_data = sc.textFile("ml-100k/u.item")
    val move_nums = move_data.count()
    println(move_nums)
    val move_first = move_data.first()
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    println(move_first)
    val move_files = move_data.map(line => line.split('|')) //以竖线分隔

    //评级数据
    val rating_data = sc.textFile("ml-100k/u.data")
    val rating_nums = rating_data.count()
    val rating_first = rating_data.first()
    //用户ID  | 影片ID   | 星级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    println("rating_first:" + rating_first)
    //println(rating_nums)
    val rating_files = rating_data.map(line=>line.split("\t")) //以竖线分隔
    val ratings = rating_files.map { line =>line(2).toInt } //注意转换Int类型
    // println(1.max(3))
    val max_rating = ratings.reduce((x, y) => x.max(y)) //最大值评级
    val min_rating = ratings.reduce((x, y) => x.min(y)) //最小值评级
    val mean_rating = ratings.reduce(_ + _)/rating_nums//平均评级
    //val median_rating=ratings.collect()
  
    val median_per_user = rating_nums / num_users //
    val retinags_per_move = rating_nums / move_nums //
    //
    val stats: StatCounter = ratings.stats()
    println("最大值:"+stats.max+"\t最小值:"+stats.min+"\t中间值:"+stats.mean+"\t总数据:"+stats.count+"\t合计值:"+stats.sum+"\t标准偏差:"+stats.stdev)
      println("统计数据:"+ratings.stats())
    println("Min rating: %d,Max rating: %d,Average rating:%d,Median rating: %d".format(min_rating, max_rating, mean_rating, median_per_user))
    //rating_data 提出用户ID为主键,评级为值的键值对
    val user_ratings_grouped=rating_data.map{file=>(file(0).toInt,file(2).toInt)}.groupByKey()
    //求出每个主键(用户ID)对应的评级集合的大小,这会给出各用户评级的次数
   // val user_ratings_byuser=user_ratings_grouped.map((k,v) => (k,count(v)))
 
    
  }

}
