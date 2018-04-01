
import scalation.linalgebra._
import scalation.analytics.classifier._
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream

object CrossValidation2 extends App{

  println("============ data science 2 Proj 2 =====================")
  val noOfFolds = args(0).toInt
  println("Number of folds is " + noOfFolds)
  var fn:Array[String] = null
  var fn1:Array[String] = null
  var cn:Array[String] = null
  var k:Integer = null
  val xy = readData(args(1))
  val size = xy._1.dim1
  val colSize = xy._1.dim2
  var metricsTb = Array(0.0 ,0.0, 0.0, 0.0, 0.0)
  var metricsNb = Array(0.0 ,0.0, 0.0, 0.0, 0.0)
  var metricsKnn = Array(0.0 ,0.0, 0.0, 0.0, 0.0)
  var metricsLgr = Array(0.0 ,0.0, 0.0, 0.0, 0.0)
  var metricsLda = Array(0.0 ,0.0, 0.0, 0.0, 0.0)
  val vc = (for (j <- xy._1.range2) yield xy._1.col(j).max() + 1).toArray


  //println("---------------------------------------------------------------")
  //println("================== Logistic Regression Classifier ====================")
  val x1 = xy._3.+^:(VectorD.one(size))
  crossValidate(xy._1,xy._2,xy._3)

  def readData(fileName:String):(MatriI, VectoI, MatrixD)={
    //val filename = "breast-cancer.arff" //args(1)
    //val filename = "caravan_small.csv" //args(1)
    var data = if(fileName.contains(".csv")) Relation (fileName, "caravan", 0, null, ",") else Relation (fileName,-1,null)
    println("Filename : " + fileName)
    //var data = Relation (fileName, "caravan", 0, null, ",")
    //val data = Relation (fileName,-1,null)
    val xy = data.toMatriI2(null)
    val size = xy.dim1
    val colSize = xy.dim2
    println(s"Data dimensions $size x $colSize")

    val X = xy.sliceCol(0,colSize)
    val xD = MatrixD(for(i<-X.range1) yield Vec.toDouble(X(i))).t
    val y = xy.col(colSize-1)

    fn = data.colName.slice(0, X.dim2 - 1).toArray
    fn1 = Array("one") ++ fn
    cn = Array ("No", "Yes")// class names
    k  = 2 // no of classes
    //(X,y,xD,fn,cn,k)
    (X,y,xD)
  }


  def split(xy:(MatriI, VectoI, MatrixD), index:Integer, itestA:Array[VectoI]) = {

    val newsize = xy._1.dim1 - itestA(index).dim
    var v:VectorI = new VectorI(newsize)
    for (i<- 0 until noOfFolds){
      if(i!= index){
       v = v.++(itestA(i))
      }
    }
    v = v.slice(newsize,v.dim) // vector of all indices except test set indices

    var testsetI = xy._1.selectRows(itestA(index).toArray)
    //println("testset "+testsetI.dim1)
    var testsety = xy._2.select(itestA(index).toArray)
    //println("testset "+testsety.dim)
    var testsetD = xy._3.selectRows(itestA(index).toArray)
    //print(v.dim)

    //val f = for (i<-0 until xy._1.dim1) diff(itestA(index))
    var trainsetI = xy._1.selectRows(v.toArray)
    //println("trainseti "+trainsetI.dim1)
    var trainsety = xy._2.select(v.toArray)
    //println("ttrainsety "+trainsety.dim)
    var trainsetD = xy._3.selectRows(v.toArray)
    //println(trainsetI)

    //x_train,  x-test,   y_train,    y_test, x_trainD, x_testD)
    (trainsetI, testsetI, trainsety, testsety, trainsetD, testsetD)
  }

  def crossValidate(xy:(MatriI,VectoI,MatrixD)):Unit={

    val permutedVec = PermutedVecI (VectorI.range(0, size), ranStream)
    val randOrder   = permutedVec.igen
    val itestA      = randOrder.split (noOfFolds)

    for (ithFold <- 0 until noOfFolds){

      println(s"Fold $ithFold")
      var traintestfold = split(xy,ithFold,itestA)

//=================================================================================================================================
      //println("---------------------------------------------------------------")
      //println("==================  Tan Bayes ====================")
      val tan = new TANBayes (traintestfold._1, traintestfold._3, fn, k, cn, vc_ = vc)      // create the classifier
      tan.train()
      tan.test(traintestfold._2,traintestfold._4)
      //print("tan cv accu = " + tan.test(traintestfold._2,traintestfold._4))
      //var yp:VectoI = tan.classify(traintestfold._2)

      var cm  = new ConfusionMat (traintestfold._4,tan.classify(traintestfold._2),k)
      var p_r = cm.prec_recl
      var fscore = cm.f1_measure(p_r._3, p_r._4)
      // precision and recall
      metricsTb(0) +=cm.accuracy // sum of accuracy across the folds
      metricsTb(1) +=p_r._3 // sum of precisions across the folds
      metricsTb(2) +=p_r._4 // sum of recall across the folds
      metricsTb(3) +=fscore // fscore across all folds
      //println(s"fit results acc ${cm.accuracy} precision ${p_r._3} recall ${p_r._4} F-score ${fscore} ")
//==================================================================================================================================
      //println("---------------------------------------------------------------")
      //println("==================  Naive Bayes ====================")
     val nb0   = new NaiveBayes0 (traintestfold._1, traintestfold._3, fn, k, cn, vc = vc)           // create the classifier
      nb0.train()
      nb0.test(traintestfold._2,traintestfold._4)

      var cm1  = new ConfusionMat (traintestfold._4,nb0.classify(traintestfold._2),k)
      var p_r1 = cm1.prec_recl
      var fscore1 = cm1.f1_measure(p_r1._3, p_r1._4)
      // precision and recall
      metricsNb(0) +=cm1.accuracy // sum of accuracy across the folds
      metricsNb(1) +=p_r1._3 // sum of precisions across the folds
      metricsNb(2) +=p_r1._4 // sum of recall across the folds
      metricsNb(3) +=fscore1 // fscore across all folds
      //println(s"fit results acc ${cm.accuracy} precision ${p_r._3} recall ${p_r._4} F-score ${fscore} ")

//=================================================================================================================================
      //println("---------------------------------------------------------------")
      //println("================== LDA Classifier ====================")
      val lda = new LDA (traintestfold._5, traintestfold._3, fn)
      lda.train()
      lda.test(traintestfold._6,traintestfold._4)
      var yplda = VectorI(for(i<-traintestfold._2.range1) yield (lda.classify(traintestfold._2(i))._1))

      var cm2  = new ConfusionMat (traintestfold._4,yplda,k)
      var p_r2 = cm2.prec_recl
      var fscore2 = cm2.f1_measure(p_r2._3, p_r2._4)
      // precision and recall
      metricsLda(0) +=cm2.accuracy // sum of accuracy across the folds
      metricsLda(1) +=p_r2._3 // sum of precisions across the folds
      metricsLda(2) +=p_r2._4 // sum of recall across the folds
      metricsLda(3) +=fscore2 // fscore across all folds
      //println(s"fit results acc ${cm.accuracy} precision ${p_r._3} recall ${p_r._4} F-score ${fscore} ")

//=================================================================================================================================
      //println("---------------------------------------------------------------")
      //println("================== KNN Classifier ====================")
      val knn = new KNN_Classifier(traintestfold._5, traintestfold._3,fn,k,cn)
      knn.train()
      knn.test(traintestfold._6,traintestfold._4)

      var ypknn = VectorI(for(i<-traintestfold._2.range1) yield (knn.classify(traintestfold._2(i))._1))
      var cm3  = new ConfusionMat (traintestfold._4,ypknn,k)
      var p_r3 = cm3.prec_recl
      var fscore3 = cm3.f1_measure(p_r3._3, p_r3._4)
      // precision and recall
      metricsKnn(0) +=cm2.accuracy // sum of accuracy across the folds
      metricsKnn(1) +=p_r3._3 // sum of precisions across the folds
      metricsKnn(2) +=p_r3._4 // sum of recall across the folds
      metricsKnn(3) +=fscore3 // fscore across all folds
      //println(s"fit results acc ${cm.accuracy} precision ${p_r._3} recall ${p_r._4} F-score ${fscore} ")

//=================================================================================================================================
      //println("---------------------------------------------------------------")
      //println("================== Logistic Regression Classifier ====================")
      val x1 = traintestfold._5.+^:(VectorD.one(size))  // prepend 1 column
      val xtest = traintestfold._6.+^:(VectorD.one(size))
      fn1 = Array("one") ++ fn
      val lgr = new LogisticRegression(x1,traintestfold._3,fn1,cn)
       lgr.train()
       lgr.test(xtest,traintestfold._4)
      var yplgr = VectorI(for(i<-traintestfold._2.range1) yield (knn.classify(traintestfold._2(i))._1))
      var cm4  = new ConfusionMat (traintestfold._4,yplgr,k)
      var p_r4 = cm4.prec_recl
      var fscore4 = cm4.f1_measure(p_r4._3, p_r4._4)
      // precision and recall
      metricsLgr(0) +=cm2.accuracy // sum of accuracy across the folds
      metricsLgr(1) +=p_r4._3 // sum of precisions across the folds
      metricsLgr(2) +=p_r4._4 // sum of recall across the folds
      metricsLgr(3) +=fscore4 // fscore across all folds
      //println(s"fit results acc ${cm.accuracy} precision ${p_r._3} recall ${p_r._4} F-score ${fscore} ")


    }


    println("---------------------------------------------------------------")
    println("==================  Tan Bayes ====================")
    println("| Acc\t\t Precision\t\t Recall\t\t F-score |")
    metricsTb = metricsTb.map(x => x/noOfFolds)
    println(s"${metricsTb(0)} | ${metricsTb(1)} | ${metricsTb(2)} | ${metricsTb(3)}")
    println("---------------------------------------------------------------")
    println("==================  Naive Bayes ====================")
    metricsNb = metricsNb.map(x1 => x1/noOfFolds)
    println(s"${metricsNb(0)} | ${metricsNb(1)} | ${metricsNb(2)} | ${metricsNb(3)}")
    println("---------------------------------------------------------------")
    println("================== LDA Classifier ====================")
    metricsLda = metricsLda.map(x => x/noOfFolds)
    println(s"${metricsLda(0)} | ${metricsLda(1)} | ${metricsLda(2)} | ${metricsLda(3)}")
    println("---------------------------------------------------------------")
    println("================== KNN Classifier ====================")
    metricsKnn = metricsKnn.map(x => x/noOfFolds)
    println(s"${metricsKnn(0)} | ${metricsKnn(1)} | ${metricsKnn(2)} | ${metricsKnn(3)}")
    println("---------------------------------------------------------------")
    println("================== Logistic Regression Classifier ====================")
    metricsLgr = metricsLgr.map(x => x/noOfFolds)
    println(s"${metricsLgr(0)} | ${metricsLgr(1)} | ${metricsLgr(2)} | ${metricsLgr(3)}")

  }

}
