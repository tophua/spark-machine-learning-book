package chapater03


import org.apache.log4j.{ Level, Logger }
import org.apache.spark.{ SparkConf, SparkContext }
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.Vectors

object breeze_test01 {

  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster("local[2]").setAppName("SparkHdfsLR")
    val sc = new SparkContext(conf)
    Logger.getRootLogger.setLevel(Level.WARN)

    // 3.1.1 Breeze 创建函数
    val m1 = DenseMatrix.zeros[Double](2, 3)
    val v1 = DenseVector.zeros[Double](3)
    val v2 = DenseVector.ones[Double](3)
    val v3 = DenseVector.fill(3) { 5.0 }
    val v4 = DenseVector.range(1, 10, 2)
    val m2 = DenseMatrix.eye[Double](3)
    //对角矩阵
    /**
     * 1.0,0.0,0.0
     * 0.0,2.0,0.0
     * 0.0,0.0,3.0
     */
    val v6 = diag(DenseVector(1.0, 2.0, 3.0))
    /**
     * 按行创建矩阵
     * 1.0 2.0
     * 3.0 4.0
     */
    val m3 = DenseMatrix((1.0, 2.0), (3.0, 4.0))
    //按行创建向量
    val v8 = DenseVector(1, 2, 3, 4)
    val v9 = DenseVector(1, 2, 3, 4).t //向量转置
    //从函数创建向量
    val v10 = DenseVector.tabulate(3) { i => 2 * i }
    val m4 = DenseMatrix.tabulate(3, 2) { case (i, j) => i + j }
    //从数组创建向量
    val v11 = new DenseVector(Array(1, 2, 3, 4))
    //从数组创建矩阵,2行3列
    val m5 = new DenseMatrix(2, 3, Array(11, 12, 13, 21, 22, 23))
    //0到1的随机向量
    val v12 = DenseVector.rand(4)
    //2行3列随机矩阵
    val m6 = DenseMatrix.rand(2, 3)

    // 3.1.2 Breeze 元素访问及操作函数
    // 元素访问
    val a = DenseVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    a(0)//指定位置
    a(1 to 4)//向量子集
    a(5 to 0 by -1)//按照指定步长取子集
    a(1 to -1)//指定开始位置至结尾
    a(-1)//最后一个元素
    //创建矩阵2行3列
    val m = DenseMatrix((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))
    m(0, 1)//取出1行,2列数据即2
    m(::, 1)//矩阵指定列,获取2列数据,即2,4

    // 创建矩阵2行3列
    val m_1 = DenseMatrix((1.0, 2.0, 3.0), (3.0, 4.0, 5.0))
    /**
     * 1.0 4.0
     * 3.0 3.0
     * 2.0 5.0
     */
    m_1.reshape(3, 2)//调整矩阵3行2列
    /**
     * 1.0,3.0,2.0,4.0,3.0,5.0
     */
    m_1.toDenseVector//把矩阵按列形式转换成向量
     // 创建矩阵3行3列
    val m_3 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    //复制下三角
    lowerTriangular(m_3)
    //复制上三角
    upperTriangular(m_3)
    //矩阵复制
    m_3.copy
    //取对象线元素
    /**
     * 1.0,20,3.0
     * 4.0,5.0,6.0
     * 7.0,8.0,9.0
     */
    diag(m_3)//取对象线元素 1.0,5.0,9.0
    //矩阵3列赋值5.0
    m_3(::, 2) := 5.0
    
    m_3
     //矩阵赋值(2行,2列向后开始赋值5.0)
    m_3(1 to 2, 1 to 2) := 5.0
    m_3
    //创建矩阵
    val a_1 = DenseVector(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    //矩阵向量赋值
    a_1(1 to 4) := 5
    a_1(1 to 4) := DenseVector(1, 2, 3, 4)
    a_1
    val a1 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val a2 = DenseMatrix((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    //垂直连接矩阵,即两个矩阵连接,列不变,行增加
    DenseMatrix.vertcat(a1, a2)
    //横向连接矩阵,行不变,列增加
    DenseMatrix.horzcat(a1, a2)
    val b1 = DenseVector(1, 2, 3, 4)
    val b2 = DenseVector(1, 1, 1, 1)
    //垂直连接矩阵,即两个矩阵连接,列不变,行增加
    DenseVector.vertcat(b1, b2)

    // 3.1.3 Breeze 数值计算函数
    val a_3 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    val b_3 = DenseMatrix((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    a_3 + b_3 //两个矩阵元素值相加
    a_3 :* b_3//两个矩阵元素值相乘
    a_3 :/ b_3//两个矩阵元素值相除
    a_3 :< b_3//两个矩阵元素值比较
    a_3 :== b_3//两个矩阵元素值相等
    a_3 :+= 1.0//元素值追加
    a_3 :*= 2.0//元素值追乘
    max(a_3)//矩阵元素最大值
    argmax(a_3)//矩阵元素最大值及位置坐标
    DenseVector(1, 2, 3, 4) dot DenseVector(1, 1, 1, 1)

    // 3.1.4 Breeze 求和函数
    val a_4 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    sum(a_4)//对矩阵求和
    sum(a_4, Axis._0)//对矩阵每列求和
    sum(a_4, Axis._1)//对矩阵每行求和
    trace(a_4)//对角线元素求和
    accumulate(DenseVector(1, 2, 3, 4))//累计和,1,3,6,10,即第一位开始向后累计求和
    

    // 3.1.5 Breeze 布尔函数
    val a_5 = DenseVector(true, false, true)
    val b_5 = DenseVector(false, true, true)
    a_5 :& b_5 //元素与
    a_5 :| b_5//元素或
    !a_5//元素非
    val a_5_2 = DenseVector(1.0, 0.0, -2.0)
    any(a_5_2)//任意元素非0
    all(a_5_2)//所有元素非0

    // 3.1.6 Breeze 线性代数函数
    val a_6 = DenseMatrix((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    val b_6 = DenseMatrix((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    a_6 \ b_6//线性求解
    a_6.t//转置 
    det(a_6)//求特征值
    inv(a_6)//求逆
    val svd.SVD(u, s, v) = svd(a_6)
    a_6.rows//矩阵行数
    a_6.cols//矩阵列数 

    // 3.1.7 Breeze 取整函数
    //矩阵向量
    val a_7 = DenseVector(1.2, 0.6, -2.3)
    round(a_7)//矩阵元素四舍五入
    ceil(a_7)//矩阵元素最小整数
    floor(a_7)//矩阵元素最大整数
    signum(a_7)//矩阵元素符号函数
    abs(a_7)//矩阵元素取正数 

  }
}