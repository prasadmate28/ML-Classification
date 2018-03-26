import scalation.linalgebra.{MatrixD, MatrixI, VectorI}
import scalation.analytics.classifier._
import scalation.columnar_db.Relation
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream

object CrossValidation extends App {

  println("============data science 2 Proj 2 =====================")
  //
  //val caravanDataMtrx = MatrixI("data/caravan.csv",1)
  //println(caravanDataMtrx.dim1 + " " + caravanDataMtrx.dim2)
  var noOfFolds = 10 // args(0)
  println("Number of folds is " + noOfFolds)

  def crossValidateRand (nx: Int = 10, obj:Classifier): Double =
  {
    var sum         = 0.0
    val permutedVec = PermutedVecI (VectorI.range(0, size), ranStream)
    val randOrder   = permutedVec.igen
    val itestA      = randOrder.split (nx)

    for (itest <- itestA) {
      obj.train (itest())
      sum += obj.test(itest)

    }
    // calc recall, precision and F score
    sum / nx.toDouble
  } // crossValidateRand

  val filename = "caravan.csv" //args(1)
  println("Filename : "+filename)
  var data = Relation (filename,"caravan", 0, null,",")
  val xy = data.toMatriI2(null)
  val size = xy.dim1
  val colSize = xy.dim2

  val X = xy.sliceCol(0,colSize)
  val y = xy.col(colSize-1)

  val fn = data.colName.slice(0, xy.dim2 - 1).toArray
  val cn = Array ("No", "Yes")// class names
  val k  = 2 // no of classes
  println("---------------------------------------------------------------")
  println("==================  Tan Bayes ====================")
  val tan0 = TANBayes0 (xy, fn, k, cn)// create the classifier
  //val tan  = TANBayes(xy, fn, k, cn)                         // create the classifier
  val acc1= crossValidateRand(10,tan0)
  println("tan  cv accu = " + acc1)        // cross validate the classifier

  println("---------------------------------------------------------------")
  println("==================  Naive Bayes ====================")
  val nb0   = NaiveBayes0 (xy, fn, k, cn)           // create the classifier
  //val nb    = NaiveBayes  (xy, fn, 2, cn, null, 0)           // create the classifier
  println ("nb0 cv accu = " + nb0.crossValidateRand ())
  //println ("nb  cv accu = " + nb.crossValidateRand ())

  println("---------------------------------------------------------------")
  println("================== KNN Classifier ====================")




}
