package com.sparkbyexamples.spark.dom

import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.sql.types.{DataTypes, DoubleType, StringType}

object DomFeatureEngineeringRunner {
  protected val numericFeatureNamesStr: String = "top-g0,left-g0,width-g0,height-g0,char-g0,txt_nd-g0,img-g0,a-g0," +
    "sibling-g0,child-g0,dep-g0,seq-g0,txt_dns-g0,pid-g0,tag-g0,nd_id-g0,nd_cs-g0,ft_sz-g0,color-g0,b_bolor-g0," +
    "rtop-g0,rleft-g0,rrow-g0,rcol-g0,dist-g0,simg-g0,mimg-g0,limg-g0,aimg-g0,saimg-g0,maimg-g0,laimg-g0," +
    "char_max-g0,char_ave-g0,own_char-g0,own_txt_nd-g0,grant_child-g0,descend-g0,sep-g0,rseq-g0,txt_nd_c-g0," +
    "vcc-g0,vcv-g0,avcc-g0,avcv-g0,hcc-g0,hcv-g0,ahcc-g0,ahcv-g0,txt_df-g0,cap_df-g0,tn_max_w-g0,tn_ave_w-g0," +
    "tn_max_h-g0,tn_ave_h-g0,a_max_w-g0,a_ave_w-g0,a_max_h-g0,a_ave_h-g0,img_max_w-g0,img_ave_w-g0,img_max_h-g0," +
    "img_ave_h-g0,tn_total_w-g0,tn_total_h-g0,a_total_w-g0,a_total_h-g0,img_total_w-g0,img_total_h-g0,top-g1," +
    "left-g1,width-g1,height-g1,char-g1,txt_nd-g1,img-g1,a-g1,sibling-g1,child-g1,dep-g1,seq-g1,txt_dns-g1," +
    "pid-g1,tag-g1,nd_id-g1,nd_cs-g1,ft_sz-g1,color-g1,b_bolor-g1,rtop-g1,rleft-g1,rrow-g1,rcol-g1,dist-g1," +
    "simg-g1,mimg-g1,limg-g1,aimg-g1,saimg-g1,maimg-g1,laimg-g1,char_max-g1,char_ave-g1,own_char-g1," +
    "own_txt_nd-g1,grant_child-g1,descend-g1,sep-g1,rseq-g1,txt_nd_c-g1,vcc-g1,vcv-g1,avcc-g1,avcv-g1,hcc-g1," +
    "hcv-g1,ahcc-g1,ahcv-g1,txt_df-g1,cap_df-g1,tn_max_w-g1,tn_ave_w-g1,tn_max_h-g1,tn_ave_h-g1,a_max_w-g1," +
    "a_ave_w-g1,a_max_h-g1,a_ave_h-g1,img_max_w-g1,img_ave_w-g1,img_max_h-g1,img_ave_h-g1,tn_total_w-g1," +
    "tn_total_h-g1,a_total_w-g1,a_total_h-g1,img_total_w-g1,img_total_h-g1,top-g2,left-g2,width-g2,height-g2," +
    "char-g2,txt_nd-g2,img-g2,a-g2,sibling-g2,child-g2,dep-g2,seq-g2,txt_dns-g2,pid-g2,tag-g2,nd_id-g2," +
    "nd_cs-g2,ft_sz-g2,color-g2,b_bolor-g2,rtop-g2,rleft-g2,rrow-g2,rcol-g2,dist-g2,simg-g2,mimg-g2,limg-g2," +
    "aimg-g2,saimg-g2,maimg-g2,laimg-g2,char_max-g2,char_ave-g2,own_char-g2,own_txt_nd-g2,grant_child-g2," +
    "descend-g2,sep-g2,rseq-g2,txt_nd_c-g2,vcc-g2,vcv-g2,avcc-g2,avcv-g2,hcc-g2,hcv-g2,ahcc-g2,ahcv-g2," +
    "txt_df-g2,cap_df-g2,tn_max_w-g2,tn_ave_w-g2,tn_max_h-g2,tn_ave_h-g2,a_max_w-g2,a_ave_w-g2,a_max_h-g2," +
    "a_ave_h-g2,img_max_w-g2,img_ave_w-g2,img_max_h-g2,img_ave_h-g2,tn_total_w-g2,tn_total_h-g2,a_total_w-g2," +
    "a_total_h-g2,img_total_w-g2,img_total_h-g2,top-g3,left-g3,width-g3,height-g3,char-g3,txt_nd-g3,img-g3," +
    "a-g3,sibling-g3,child-g3,dep-g3,seq-g3,txt_dns-g3,pid-g3,tag-g3,nd_id-g3,nd_cs-g3,ft_sz-g3,color-g3," +
    "b_bolor-g3,rtop-g3,rleft-g3,rrow-g3,rcol-g3,dist-g3,simg-g3,mimg-g3,limg-g3,aimg-g3,saimg-g3,maimg-g3," +
    "laimg-g3,char_max-g3,char_ave-g3,own_char-g3,own_txt_nd-g3,grant_child-g3,descend-g3,sep-g3,rseq-g3," +
    "txt_nd_c-g3,vcc-g3,vcv-g3,avcc-g3,avcv-g3,hcc-g3,hcv-g3,ahcc-g3,ahcv-g3,txt_df-g3,cap_df-g3,tn_max_w-g3," +
    "tn_ave_w-g3,tn_max_h-g3,tn_ave_h-g3,a_max_w-g3,a_ave_w-g3,a_max_h-g3,a_ave_h-g3,img_max_w-g3,img_ave_w-g3," +
    "img_max_h-g3,img_ave_h-g3,tn_total_w-g3,tn_total_h-g3,a_total_w-g3,a_total_h-g3,img_total_w-g3,img_total_h-g3"

  protected val numericFeatureNames: Array[String] = numericFeatureNamesStr.split(",")

  protected val textFeatureNamesStr = "tag,id,class,color,bg-color,font,text,url"
  protected val textFeatureNames: Array[String] = textFeatureNamesStr.split(",")

  // 创建一个 UDF 来放大向量中的前3个元素
  def scaleFirstThreeElements(vector: org.apache.spark.ml.linalg.Vector): org.apache.spark.ml.linalg.Vector = {
    val array = vector.toArray
    array(0) *= 10
    array(1) *= 10
    array(2) *= 10
    Vectors.dense(array)
  }

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("SparkByExamples.com")
      .getOrCreate()

    val filePath = "src/main/resources/csv/amazon.dataset.1.4.0-300.csv"

    var structField = DataTypes.createStructField("label", StringType, true)
    val a = numericFeatureNames.map(name => DataTypes.createStructField(name, DoubleType, true))
    val b = textFeatureNames.map(name => DataTypes.createStructField(name, DataTypes.StringType, true))
    val structureFields = Array(structField) ++ a ++ b

    val schema = DataTypes.createStructType(structureFields)

    var df = spark.read.option("header", value = true).schema(schema).csv(filePath)
    df.show(false)

    val assembler = new VectorAssembler().setInputCols(numericFeatureNames).setOutputCol("numericFeatures")
    df = assembler.transform(df)

    val scaler = new StandardScaler().setInputCol("numericFeatures").setOutputCol("scaledFeatures")
    // Compute summary statistics by fitting the StandardScaler
    val scalerModel = scaler.fit(df)
    // Normalize each feature to have unit standard deviation
    df = scalerModel.transform(df)

    df.select("scaledFeatures", "tag", "class", "url").show()
    df.select("scaledFeatures").printSchema()

    val numericFeatures = df.select("numericFeatures").rdd
      .map(r => r.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(x => x.toArray.apply(0))
    numericFeatures.take(10).foreach(println)

    df.select("scaledFeatures").rdd.take(10)
      .map(r => r.getAs[org.apache.spark.ml.linalg.Vector](0))
      .foreach(println)

    df.select("scaledFeatures").show()

    val scaleFirstThreeElementsUDF = functions.udf(scaleFirstThreeElements _)
    val scaledDfWithScaledElements = df
      .withColumn("scaledFeatures", scaleFirstThreeElementsUDF(df("scaledFeatures")))

    scaledDfWithScaledElements.select("scaledFeatures").show()
  }
}
