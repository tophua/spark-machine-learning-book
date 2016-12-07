import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.numericRDDToDoubleRDDFunctions

object Chapter05 {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(328); 
  println("Welcome to the Scala worksheet");$skip(77); 
    /**用户数据****/
    val rawData = sc.textFile("ml-100k/train_noheader.tsv");System.out.println("""rawData  : <error> = """ + $show(rawData ));$skip(56); 
    val records = rawData.map(line => line.split("\t"));System.out.println("""records  : <error> = """ + $show(records ));$skip(19); val res$0 = 
     records.first;System.out.println("""res0: <error> = """ + $show(res$0));$skip(990); 
     
     
     //由于数据格式的问题，我们做一些数据清理的工作：把额外的(“)去掉，。
    val data = records.map { r =>
      //把额外的(“)去掉
      val trimmed = r.map { x =>
        //println("befor:" + x)
        val v = x.replaceAll("\"", "")
        // println("after:" + v)
        v
      }
      //println("r:" + r.toList + "\t size:" + r.size)
      //r是Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...
      //数据一条数据为0开始
      val label = trimmed(r.size - 1).toInt
      println("label:" + label)
      //创建一个迭代器返回由这个迭代器所产生的值区间,取子集set(1,4为元素位置, 从0开始),从位置4开始,到数组的长度
      //slice提取第五列到25列的特征矩阵
      //数据集中缺失数据为?,直接用0替换缺失数据
      val features = trimmed.slice(4, r.size - 1).map{
        d => if (d == "?"){
          println(d)
          0.0
         
        }else{
           d.toDouble
        }
      }
      //将标签和特征向量转换为LabeledPoint为实例,将特征向量存储到MLib的Vectors中
      LabeledPoint(label, Vectors.dense(features))
    };System.out.println("""data  : <error> = """ + $show(data ))}
}
