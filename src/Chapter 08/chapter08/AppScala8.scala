package chapter08

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import breeze.linalg.csvwrite
import java.awt.image.BufferedImage
/**
 * 从数据中抽取合适的数据,降维模型
 */
object AppScala8 {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test").setMaster("local")
    val sc = new SparkContext(conf)
    //使用通配符的路径标识来告诉Spark在lfw文件夹中访问每个文件夹以获得文件
    val path = "lfw/*"
    //wholeTextFiles 读取一次整个文件,textFile在一个文件或多个文件逐行读取数据  
    val rdd = sc.wholeTextFiles(path)
    val first = rdd.first
    println(first)
    //本地第一步先把硬盘目录替换掉
    //再把file:替换掉
    val files = rdd.map { case (fileName, content) => fileName.replace("file:", "") }
    println(files.first)
    println(files.count)
    /**加载图片**/
    import java.awt.image.BufferedImage
    //从文件中读取图片
    def loadImageFromFile(path: String): BufferedImage = {
      import javax.imageio.ImageIO
      import java.io.File
      ImageIO.read(new File(path))
    }
    //val aePath = "file:/D:/spark/spark-1.5.0-hadoop2.6/bin/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
    val aePath1 = "lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"
 
    val aeImage = loadImageFromFile(aePath1)
    /**
     * aeImage: type = 5 ColorModel: #pixelBits = 24 numComponents = 3 
     *  color space = java.awt.color.ICC_ColorSpace@4edeb425 transparency = 1 
     *  has alpha = false isAlphaPre = false 
     *  ByteInterleavedRaster: width = 250 height = 250 #numDataElements 3 dataOff[0] = 2
     *  图片高度和宽度都250像素,颜色组件(RGB)数为3,
     */
    //转换灰度图片并改变图片大小 
    def processImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
      //TYPE_BYTE_GRAY表示无符号byte灰度级图像（无索引）
      val bwImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      val g = bwImage.getGraphics() //为组件创建一个图形上下文
      /**
       * img - 要绘制的指定图像。如果 img 为 null，则此方法不执行任何动作。
       * x - x 坐标。
       * y - y 坐标。
       * width - 矩形的宽度。
       * height - 矩形的高度。
       * observer - 当转换了更多图像时要通知的对象。
       */
      g.drawImage(image, 0, 0, width, height, null) //绘制指定图像中当前可用的图像
      //释放此图形的上下文以及它使用的所有系统资源。
      g.dispose()
      bwImage
    }
     //转换灰度图片并改变图片大小 
    val grayImage = processImage(aeImage, 100, 100)

    // write the image out to the file system
    import javax.imageio.ImageIO
    import java.io.File
    //输出图片位置
    ImageIO.write(grayImage, "jpg", new File("/tmp/aeGray.jpg"))

    // extract the raw pixels from the image as a Double array
    def getPixelsFromImage(image: BufferedImage): Array[Double] = {
      val width = image.getWidth
      val height = image.getHeight
      //用来声明多维数组
      val pixels = Array.ofDim[Double](width * height)
      
      /**
       * getPixels  读取像素的数值，存到pixels数组里面
       *     参数1 从位图中读取的第一个像素的x坐标值
       *     参数2 从位图中读取的第一个像素的y坐标值
       *     参数3 从每一行中读取的像素宽度 
       *     参数4 读取的行数 
       *     参数5 接收位图颜色值的数组 
       */
       image.getData.getPixels(0, 0, width, height, pixels)
      // pixels.map(p => p / 255.0) 		// optionally scale to [0, 1] domain
    }
    // put all the functions together
    //接受一个图片文件位置和需要处理的宽和高,返回一个包含像素数据的Array[Double]值
    def extractPixels(path: String, width: Int, height: Int): Array[Double] = {
      val raw = loadImageFromFile(path)
      val processed = processImage(raw, width, height)
      getPixelsFromImage(processed)
    }
    //把每个函数应用到包含图片路径的RDD的每个元素上将 产生一个新RDD,新的RDD包含第张图片的像素数据
    val pixels = files.map(f => extractPixels(f, 50, 50))
    println(pixels.take(10).map(_.take(10).mkString("", ",", ", ...")).mkString("\n"))

    // create vectors
    //每个图片创建MLlib对象,将缓存RDD来加速后计算 
    import org.apache.spark.mllib.linalg.Vectors
    val vectors = pixels.map(p => Vectors.dense(p))
    // the setName method createa a human-readable name that is displayed in the Spark Web UI
    vectors.setName("image-vectors")
    // remember to cache the vectors to speed up computation
    vectors.cache
    //正则化,提取列的平均值 
    // normalize the vectors by subtracting the column means
    import org.apache.spark.mllib.feature.StandardScaler
    //fit函数会导致基于RDD计算
    val scaler = new StandardScaler(withMean = true, withStd = false).fit(vectors)
    //将返回的scaler来转换原始的图片向量,让所有向量减去当前列的平均值
    val scaledVectors = vectors.map(v => scaler.transform(v))
    // create distributed RowMatrix from vectors, and train PCA on it
   //PAC和SVD的计算是通过提供基于RowMatrix的方法实现
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    //我们已经从图像数据中提取出了向量,
    val matrix = new RowMatrix(scaledVectors)
    val K = 10
    //初始化一个新RowMatrix,并调用computePrincipalComponents来计算分布矩阵的前K个主成分
    val pc = matrix.computePrincipalComponents(K)
    // you may see warnings, if the native BLAS libraries are not installed, don't worry about these 
    // 14/09/17 19:53:49 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    // 14/09/17 19:53:49 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK

    // use Breeze to save the principal components as a CSV file
    //主成分析行业和列数
    val rows = pc.numRows
    val cols = pc.numCols
    println(rows, cols)
    // (2500,10)
    import breeze.linalg.DenseMatrix
    //
    val pcBreeze = new DenseMatrix(rows, cols, pc.toArray)
    import breeze.linalg.csvwrite
    import java.io.File
    //csvwrite 把矩阵写到CSV文件中
    csvwrite(new File("/tmp/pc.csv"), pcBreeze)

    // project the raw images to the K-dimensional space of the principla components
    //矩阵乘法把图像矩阵和主成分析矩阵相乘实现投影矩阵
    val projected = matrix.multiply(pc)
    println(projected.numRows, projected.numCols)
    // (1055,10)
    println(projected.rows.take(5).mkString("\n"))
    /*
    [2648.9455749636277,1340.3713412351376,443.67380716760965,-353.0021423043161,52.53102289832631,423.39861446944354,413.8429065865399,-484.18122999722294,87.98862070273545,-104.62720604921965]
    [172.67735747311974,663.9154866829355,261.0575622447282,-711.4857925259682,462.7663154755333,167.3082231097332,-71.44832640530836,624.4911488194524,892.3209964031695,-528.0056327351435]
    [-1063.4562028554978,388.3510869550539,1508.2535609357597,361.2485590837186,282.08588829583596,-554.3804376922453,604.6680021092125,-224.16600191143075,-228.0771984153961,-110.21539201855907]
    [-4690.549692385103,241.83448841252638,-153.58903325799685,-28.26215061165965,521.8908276360171,-442.0430200747375,-490.1602309367725,-456.78026845649435,-78.79837478503592,70.62925170688868]
    [-2766.7960144161225,612.8408888724891,-405.76374113178616,-468.56458995613974,863.1136863614743,-925.0935452709143,69.24586949009642,-777.3348492244131,504.54033662376435,257.0263568009851]
		*/

    // relationship to SVD
    //主成分析和奇异值的关系
    val svd = matrix.computeSVD(10, computeU = true)
    //SVD计算产生的右奇异向量等于我们计算得到的主成分
    println(s"U dimension: (${svd.U.numRows}, ${svd.U.numCols})")
    println(s"S dimension: (${svd.s.size}, )")
    println(s"V dimension: (${svd.V.numRows}, ${svd.V.numCols})")
    // U dimension: (1055, 10),U矩阵等于投影数据
    // S dimension: (10, )
    // V dimension: (2500, 10)  右奇异向量等于我们计算得到的主成分
    // simple function to compare the two matrices, with a tolerance for floating point number comparison
    //比较两个向量数据
    def approxEqual(array1: Array[Double], array2: Array[Double], tolerance: Double = 1e-6): Boolean = {
      // note we ignore sign of the principal component / singular vector elements
      val bools = array1.zip(array2).map { case (v1, v2) => if (math.abs(math.abs(v1) - math.abs(v2)) > 1e-6) false else true }
      bools.fold(true)(_ & _)//fold函数开始对传进的两个参数进行计算,初始值为true,& 从左到右
    }
    // test the function
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(1.0, 2.0, 3.0)))
    // true
    println(approxEqual(Array(1.0, 2.0, 3.0), Array(3.0, 2.0, 1.0)))
    // false
    println(approxEqual(svd.V.toArray, pc.toArray))
    // true

    // compare projections
    val breezeS = breeze.linalg.DenseVector(svd.s.toArray)
    val projectedSVD = svd.U.rows.map { v =>
      val breezeV = breeze.linalg.DenseVector(v.toArray)
      val multV = breezeV :* breezeS
      Vectors.dense(multV.data)
    }
    projected.rows.zip(projectedSVD).map { case (v1, v2) => approxEqual(v1.toArray, v2.toArray) }.filter(b => true).count
    // 1055

    // inspect singular values
    //评估SVD的k值
    val sValues = (1 to 5).map { i => matrix.computeSVD(i, computeU = false).s }
    sValues.foreach(println)
    /*
    [54091.00997110354]
    [54091.00997110358,33757.702867982436]
    [54091.00997110357,33757.70286798241,24541.193694775946]
    [54091.00997110358,33757.70286798242,24541.19369477593,23309.58418888302]
    [54091.00997110358,33757.70286798242,24541.19369477593,23309.584188882982,21803.09841158358]
		*/
    val svd300 = matrix.computeSVD(300, computeU = false)
    val sMatrix = new DenseMatrix(1, 300, svd300.s.toArray)
    csvwrite(new File("/tmp/s.csv"), sMatrix)
  }
}
