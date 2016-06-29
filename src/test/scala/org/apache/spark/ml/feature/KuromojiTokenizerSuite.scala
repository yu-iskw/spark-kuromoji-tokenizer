/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import java.io.File

import scala.beans.BeanInfo

import org.atilika.kuromoji.{Tokenizer => KTokenizer}

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.functions.lit

@BeanInfo
case class TestText(text: String)

class KuromojiTokenizerSuite extends SparkFunSuite with MLlibTestSparkContext {

  private val data = Seq(
    TestText("天皇は、日本国の象徴であり日本国民統合の象徴であつて、この地位は、主権の存する日本国民の総意に基く。"),
    TestText("皇位は、世襲のものであつて、国会の議決した皇室典範 の定めるところにより、これを継承する。"),
    TestText("天皇の国事に関するすべての行為には、内閣の助言と承認を必要とし、内閣が、その責任を負ふ。"),
    TestText("天皇は、この憲法の定める国事に関する行為のみを行ひ、国政に関する権能を有しない。"),
    TestText("天皇は、法律の定めるところにより、その国事に関する行為を委任することができる。"),
    TestText("")
  )

  test("transform") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd)

    val kuromoji = new KuromojiTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setMode("EXTENDED")
    assert(kuromoji.getMode === "EXTENDED")

    val transformed = kuromoji.transform(df)
    val tokensList = transformed.select("tokens").collect()
    assert(tokensList(0).getSeq(0).size === 32)
    assert(tokensList(1).getSeq(0).size === 28)
    assert(tokensList(2).getSeq(0).size === 29)
    assert(tokensList(3).getSeq(0).size === 23)
    assert(tokensList(4).getSeq(0).size === 20)
    assert(tokensList(5).getSeq(0).size === 0)
  }

  test("pipeline and corss-validation") {
    val rdd = sc.parallelize(data)
    val df = sqlContext.createDataFrame(rdd).withColumn("label", lit(1.0))

    val kuromoji = new KuromojiTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setMode("EXTENDED")
    val hashingTF = new HashingTF()
      .setInputCol(kuromoji.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setLabelCol("label")
    val pipeline = new Pipeline()
        .setStages(Array(kuromoji, hashingTF, lr))
    val model = pipeline.fit(df)
    val transformed = model.transform(df)
    val tokensList = transformed.select("tokens").collect()
    assert(tokensList(0).getSeq(0).size === 32)
    assert(tokensList(1).getSeq(0).size === 28)
    assert(tokensList(2).getSeq(0).size === 29)
    assert(tokensList(3).getSeq(0).size === 23)
    assert(tokensList(4).getSeq(0).size === 20)
    assert(tokensList(5).getSeq(0).size === 0)

    val paramGrid = new ParamGridBuilder()
      .addGrid(kuromoji.mode, Array("NORMAL", "SEARCH", "EXTENDED"))
      .build()
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
    val cvModel = cv.fit(df)
    val transformed2 = cvModel.transform(df)
    val tokensList2 = transformed2.select("tokens").collect()
    assert(tokensList(0).getSeq(0).size === 32)
    assert(tokensList(1).getSeq(0).size === 28)
    assert(tokensList(2).getSeq(0).size === 29)
    assert(tokensList(3).getSeq(0).size === 23)
    assert(tokensList(4).getSeq(0).size === 20)
    assert(tokensList(5).getSeq(0).size === 0)
  }

  test("save/load") {
    val kuromoji = new KuromojiTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setMode("EXTENDED")
    val path = File.createTempFile("spark-kuromoji-tokenizer", "").getAbsolutePath
    kuromoji.write.overwrite().save(path)
    val loadedModel = KuromojiTokenizer.load(path)
    assert(loadedModel.getMode === "EXTENDED")
  }
}
