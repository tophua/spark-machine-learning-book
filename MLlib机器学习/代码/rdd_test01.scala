import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.SparkContext._

object rdd_test01 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("rdd_test01")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 2.1节
    // 2.1.1 RDD 创建操作
    val data = Array(1, 2, 3, 4, 5, 6, 7, 8, 9)
    val distData = sc.parallelize(data, 3)

    val distFile1 = sc.textFile("data.txt") // 本地当前目录下的文件
    val distFile2 = sc.textFile("hdfs://192.168.1.100:9000/input/data.txt") // HDFS 文件
    val distFile3 = sc.textFile("file:/input/data.txt") // 本地指定目录下的文件
    val distFile4 = sc.textFile("/input/data.txt") // 本地指定目录下的文件

    // 2.1.2 RDD 转换操作
    val rdd1 = sc.parallelize(1 to 9, 3)
    val rdd2 = rdd1.map(x => x * 2)
    rdd2.collect

    val rdd3 = rdd2.filter(x => x > 10)
    rdd3.collect
    val rdd4 = rdd3.flatMap(x => x to 20)

    val rdd5 = rdd1.mapPartitions(myfunc)
    rdd5.collect

    val a = sc.parallelize(1 to 10000, 3)
    a.sample(false, 0.1, 0).count

    val rdd8 = rdd1.union(rdd3)
    rdd8.collect

    val rdd9 = rdd8.intersection(rdd1)
    rdd9.collect

    val rdd10 = rdd8.union(rdd9).distinct
    rdd10.collect

    val rdd0 = sc.parallelize(Array((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)), 3)
    val rdd11 = rdd0.groupByKey()
    rdd11.collect

    val rdd12 = rdd0.reduceByKey((x, y) => x + y)
    rdd12.collect

    // val z = sc.parallelize(List(1, 2, 3, 4, 5, 6), 2)
    //z.aggregate(0)(math.max(_, _),  _ + _)
    val z = sc.parallelize(List((1, 3), (1, 2), (1, 4), (2, 3)))
    z.aggregateByKey(0)(math.max(_, _), _ + _).collect

    val data2 = Array((1, 1.0), (1, 2.0), (1, 3.0), (2, 4.0), (2, 5.0), (2, 6.0))
    val rdd = sc.parallelize(data2, 2)
    val combine1 = rdd.combineByKey(createCombiner = (v: Double) => (v: Double, 1),
      mergeValue = (c: (Double, Int), v: Double) => (c._1 + v, c._2 + 1),
      mergeCombiners = (c1: (Double, Int), c2: (Double, Int)) => (c1._1 + c2._1, c1._2 + c2._2),
      numPartitions = 2)
    combine1.collect

    val rdd14 = rdd0.sortByKey()
    rdd14.collect

    val rdd15 = rdd0.join(rdd0)
    rdd15.collect

    val rdd16 = rdd0.cogroup(rdd0)
    rdd16.collect

    val rdd17 = rdd1.cartesian(rdd3)
    rdd17.collect

    val rdd18 = sc.parallelize(1 to 9, 3)
    rdd18.pipe("head -n 1").collect

    val rdd19 = rdd1.randomSplit(Array(0.3, 0.7), 1)
    rdd19(0).collect
    rdd19(1).collect

    val rdd20_1 = sc.parallelize(1 to 9, 3)
    val rdd20_2 = sc.parallelize(1 to 3, 3)
    val rdd20_3 = rdd20_1.subtract(rdd20_2)

    val rdd21_1 = sc.parallelize(Array(1, 2, 3, 4), 3)
    val rdd21_2 = sc.parallelize(Array("a", "b", "c", "d"), 3)
    val rdd21_3 = rdd21_1.zip(rdd21_2)

    val data3 = Array((1, 1.0), (1, 2.0), (1, 3.0), (2, 4.0), (2, 5.0), (2, 6.0))
    val rdd24 = sc.parallelize(data3, 2)
    val combine24_1 = rdd24.combineByKey(createCombiner = (v: Double) => (v: Double, 1),
      mergeValue = (c: (Double, Int), v: Double) => (c._1 + v, c._2 + 1),
      mergeCombiners = (c1: (Double, Int), c2: (Double, Int)) => (c1._1 + c2._1, c1._2 + c2._2),
      numPartitions = 2)
    val treeAggregate24_1 = rdd24.treeAggregate((0, 0.0))(seqOp = ((u, t) => (u._1 + t._1, u._2 + t._2)),
      combOp = (u1, u2) => (u1._1 + u2._1, u1._2 + u2._2),
      depth = 2)

    // 2.1.3 RDD 行动操作
    val rdd3_1 = sc.parallelize(1 to 9, 3)
    val rdd3_2 = rdd3_1.reduce(_ + _)
    rdd3_1.collect()
    rdd3_1.count()
    rdd3_1.first()
    rdd3_1.take(3)
    rdd3_1.takeSample(true, 4)
    rdd3_1.takeOrdered(4)

  }

  def myfunc[T](iter: Iterator[T]): Iterator[(T, T)] = {
    var res = List[(T, T)]()
    var pre = iter.next
    while (iter.hasNext) {
      val cur = iter.next
      res.::=(pre, cur)
      pre = cur
    }
    res.iterator
  }

}