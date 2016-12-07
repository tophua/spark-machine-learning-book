package chapater03

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext.doubleRDDToDoubleRDDFunctions
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.util.StatCounter
import breeze.linalg.max
import breeze.linalg.min
import scala.collection.mutable.TreeSet
import scala.collection.immutable.TreeMap
import scala.collection.immutable.HashMap
import scala.collection.mutable

/**
 * A simple Spark app in Scala
 */
object ScalaApp3 {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(sparkConf)
    /**用户数据****/
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
    /**评级数据**/
    val rating_data = sc.textFile("ml-100k/u.data")
    val rating_nums = rating_data.count()
    val rating_first = rating_data.first()
    //用户ID  | 影片ID   | 星级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    println("rating_first:" + rating_first)
    //println(rating_nums)
    val rating_files = rating_data.map(line => line.split("\t")) //以竖线分隔
    val ratings = rating_files.map { line => line(2).toInt } //注意转换Int类型
    // println(1.max(3))
    val max_rating = ratings.reduce((x, y) => x.max(y)) //最大值评级
    val min_rating = ratings.reduce((x, y) => x.min(y)) //最小值评级
    val mean_rating = ratings.reduce(_ + _) / rating_nums //平均评级
    //val median_rating=ratings.collect()

    val median_per_user = rating_nums / num_users //评级总数/用户总数
    val retinags_per_move = rating_nums / move_nums //评级总数/电影总数
    //
    val stats: StatCounter = ratings.stats()
    println("最大值:" + stats.max + "\t最小值:" + stats.min + "\t中间值:" + stats.mean + "\t总数据:" + stats.count + "\t合计值:" + stats.sum + "\t标准偏差:" + stats.stdev)
    println("统计数据:" + ratings.stats())
    println("Min rating: %d,Max rating: %d,Average rating:%d,Median rating: %d".format(min_rating, max_rating, mean_rating, median_per_user))
    //rating_data 提出用户ID为主键,评级为值的键值对
    val user_ratings_grouped = rating_data.map { file => (file(0).toInt, file(2).toInt) }.groupByKey()
    //求出每个主键(用户ID)对应的评级集合的大小,这会给出各用户评级的次数
    // val user_ratings_byuser=user_ratings_grouped.map((k,v) => (k,count(v)))

    /***电影数据***/
    val movie_data = sc.textFile("ml-100k/u.item")
    //电影ID|电影标题                       |发行时间          |
    //1     |Toy Story (1995)|01-Jan-1995||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0
    println(movie_data.first())
    //总数据1682s
    val num_movies = movie_data.count()
    println("Movies: %d".format(num_movies))
    val movie_fields = movie_data.map(lines => lines.split('|')) //注意是单引号 

    //从电影数据中获取第三个字段并取出年份并转成数值,如果不匹配默认为1900年 
    val pattern = "([0-9]+)-([A-Za-z]+)-([0-9]+)".r
    val years = movie_fields.map(fields => fields(2)).map(x => {
      x match {
        case pattern(num, item, year) => {
          //println(">>>>>>>>>>>"+year.toInt)
          year.toInt
        }
        case _ => {
          //println(x)
          1900
        }
      }
    })
    // # we filter out any 'bad' data points here
    //过滤掉1900
    val years_filtered = years.filter(x => x != 1900)
    val years_1900 = years.filter(x => x == 1900)
    //使用1988年减去年份,同时按K类型统计,包含<K,Long>键值对，Long是每个K出现的频率
    //计算不同年龄电影数目
    val movie_ages = years_filtered.map(yr => 1998 - yr).countByValue()
    //取出电影数目
    val values = movie_ages.values
    //取出电影年龄
    val bins = movie_ages.keys

    /**评级数据**/
    val rating_data_raw = sc.textFile("ml-100k/u.data")
    //用户ID  | 影片ID   | 评级   | 时间戳	
    //196	    | 242	    |  3   |	881250949
    println(rating_data_raw.first())
    //10W条数据
    val num_ratings = rating_data_raw.count()
    println("等级总数据:" + num_ratings)
    //数据分隔
    val rating_dataRaw = rating_data_raw.map(line => line.split("\t"))
    //取出等级数据
    val ratingsRaw = rating_dataRaw.map(fields => fields(2).toInt)
    //取出最大等级
    val max_ratingRaw = ratingsRaw.reduce((x, y) => max(x, y))
    //取出最小等级
    val min_ratingRaw = ratingsRaw.reduce((x, y) => min(x, y))
    //平均等级3.5
    val mean_ratingRaw = ratingsRaw.reduce((x, y) => x + y) / num_ratings.toFloat
    val statsRaw: StatCounter = ratings.stats()
    //评级中位数4
    val median_rating = statsRaw.mean
    //平均用户评级
    val ratings_per_user = num_ratings / num_users
    //平均电影评级
    val ratings_per_movie = num_ratings / num_movies
    print("Min rating: %d".format(min_rating))
    print("Max rating: %d".format(max_rating))
    print("Average rating: %2.2f".format(mean_rating))
    print("Median rating: %d".format(median_rating))
    print("Average # of ratings per user: %2.2f".format(ratings_per_user))
    print("Average # of ratings per movie: %2.2f".format(ratings_per_movie))

    /**缺失数据填充对下面发行日期有问题进行填充**/
    //过滤掉发行日期有问题 
    val years_pre_processed = movie_fields.map(fields => fields(2)).map(x => {
      x match {
        case pattern(num, item, year) => {
          //println(">>>>>>>>>>>"+year.toInt)
          year.toInt
        }
        case _ => {
          //println(x)
          1900
        }
      }
    }).filter(yr => yr != 1900)

    val years_pre_processed_array: StatCounter = years_pre_processed.stats()

    /**职业**/
    val user_fields = user_data.map(line => line.split('|'))
    //用户ID|年龄  |性别|   职业         | 邮编
    // 1    |24  |  M |technician| 85711
    val all_occupations = user_fields.map(fields => fields(3)).distinct().collect()
    //排序
    all_occupations.sorted
    val map = mutable.Map.empty[String, Int]
    var idx = 0
    for (occu <- all_occupations) {
      idx += 1
      map(occu) = idx
    }
    println("Encoding of 'doctor': %d".format(map("doctor")))
    println("Encoding of 'programmer': %d".format(map("programmer")))

  }

}
