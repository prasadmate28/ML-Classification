import scalation.linalgebra.{MatrixD, MatrixI,VectorD, VectoI, VectorI}
import scalation.analytics.classifier._
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream

object CrossValidation extends App {

  println("============ data science 2 Proj 2 =====================")
  //
  var noOfFolds = 5 // args(0)
  println("Number of folds is " + noOfFolds)

  def classify(itest: VectoI, obj:Classifier):Unit = {

    println("inside classify")
    for (i<-itest){
      val yp = obj.classify(X(i))
      val yo = y(i)
      //print(s"prediction yp $yp")
    }

  }

  def crossValidateRand(nx: Int = 10, obj:Classifier): Double =
  {
    println("inside cross validate")
    var sum         = 0.0
    val permutedVec = PermutedVecI (VectorI.range(0, size), ranStream)
    val randOrder   = permutedVec.igen
    val itestA      = randOrder.split (nx)

    for (itest <- itestA) {
      obj.train (itest())
      var acc = obj.test(itest)

      sum += acc
      //classify(itest,obj)

    }
    // calc recall, precision and F score
    sum / nx.toDouble
  } // crossValidateRand

  val filename = "caravan_small.csv" //args(1)
  println("Filename : "+filename)
  var data = Relation (filename,"caravan", 0, null,",")
  val xy = data.toMatriI2(null)
  val size = xy.dim1
  val colSize = xy.dim2

  val xD = data.toMatriD(0 to colSize-2)

  val X = xy.sliceCol(0,colSize)
  val y = xy.col(colSize-1)

  val fn = data.colName.slice(0, X.dim2).toArray
  val cn = Array ("No", "Yes")// class names
  val k  = 2 // no of classes
  println("---------------------------------------------------------------")
  println("==================  Tan Bayes ====================")
  /*val tan = new TANBayes0 (X, y, fn, k, cn)      // create the classifier
  val acc1= crossValidateRand(noOfFolds,tan)
  println("tan  cv accu = " + acc1)*/         // cross validate the classifier

  println("---------------------------------------------------------------")
  println("==================  Naive Bayes ====================")
 /* val nb0   = new NaiveBayes0 (X, y, fn, k, cn)           // create the classifier
  val acc2 = crossValidateRand (noOfFolds,nb0)

  println ("nb0 cv accu = " + acc2)*/

  println("---------------------------------------------------------------")
  println("================== KNN Classifier ====================")
  val knn = new KNN_Classifier(xD,y,fn,k,cn)
  crossValidateRand(noOfFolds,knn)

  println("---------------------------------------------------------------")
  println("================== LDA Classifier ====================")
  /*val lda = new LDA (xD.asInstanceOf[MatrixD], y, fn)
  crossValidateRand(noOfFolds,lda)*/

  println("---------------------------------------------------------------")
  println("================== Logistic Regression Classifier ====================")
  /*val x1 = xD.+^:(VectorD.one(size))  // prepend 1 column
  val lgr = new LogisticRegression(x1,y,fn,cn)
  crossValidateRand(noOfFolds,lgr)*/


}
