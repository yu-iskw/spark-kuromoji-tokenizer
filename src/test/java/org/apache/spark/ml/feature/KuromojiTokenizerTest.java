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

package org.apache.spark.ml.feature;


import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class KuromojiTokenizerTest {
  private transient JavaSparkContext jsc;
  private transient SQLContext jsql;

  private List<Row> data = Arrays.asList(
      RowFactory.create("天皇は、日本国の象徴であり日本国民統合の象徴であつて、この地位は、主権の存する日本国民の総意に基く。"),
      RowFactory.create("皇位は、世襲のものであつて、国会の議決した皇室典範 の定めるところにより、これを継承する。"),
      RowFactory.create("天皇の国事に関するすべての行為には、内閣の助言と承認を必要とし、内閣が、その責任を負ふ。"),
      RowFactory.create("天皇は、この憲法の定める国事に関する行為のみを行ひ、国政に関する権能を有しない。"),
      RowFactory.create("天皇は、法律の定めるところにより、その国事に関する行為を委任することができる。"),
      RowFactory.create("")
  );

  @Before
  public void setUp() {
    jsc = new JavaSparkContext("local", "JavaKuromojiTokenizerSuite");
    jsql = new SQLContext(jsc);
  }

  @After
  public void tearDown() {
    jsc.stop();
    jsc = null;
  }

  @Test
  public void testRun() {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("text", DataTypes.StringType, false)});
    DataFrame df = jsql.createDataFrame(rdd, schema);

    KuromojiTokenizer tokenizer = new KuromojiTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setMode("EXTENDED");
    DataFrame transformed = tokenizer.transform(df);
    List<Row> tokensList = transformed.select("tokens").collectAsList();
    assertEquals(tokensList.get(0).getList(0).size(), 32);
    assertEquals(tokensList.get(1).getList(0).size(), 28);
    assertEquals(tokensList.get(2).getList(0).size(), 29);
    assertEquals(tokensList.get(3).getList(0).size(), 23);
  }

  @Test
  public void testPipeline() {
    JavaRDD<Row> rdd = jsc.parallelize(data);
    StructType schema = DataTypes.createStructType(new StructField[]{
        DataTypes.createStructField("text", DataTypes.StringType, false)});
    DataFrame df = jsql.createDataFrame(rdd, schema);
    jsql.registerDataFrameAsTable(df, "df");
    DataFrame df2 = jsql.sql("SELECT text, cast(1.0 as double) AS label FROM df");

    KuromojiTokenizer tokenizer = new KuromojiTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setMode("EXTENDED");
    HashingTF hashingTF = new HashingTF()
        .setNumFeatures(1000)
        .setInputCol(tokenizer.getOutputCol())
        .setOutputCol("features");
    LogisticRegression lr = new LogisticRegression()
        .setLabelCol("label")
        .setMaxIter(10)
        .setRegParam(0.01);
    Pipeline pipeline = new Pipeline()
        .setStages(new PipelineStage[] {tokenizer, hashingTF, lr});

    ParamMap[] paramGrid = new ParamGridBuilder().build();
    CrossValidator cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEstimatorParamMaps(paramGrid)
        .setEvaluator(new BinaryClassificationEvaluator())
        .setNumFolds(2);
    CrossValidatorModel cvModel = cv.fit(df2);
    DataFrame transformed = cvModel.transform(df2);
    Row[] tokensList = transformed.select("tokens").collect();
    assertEquals(tokensList[0].getList(0).size(), 32);
    assertEquals(tokensList[1].getList(0).size(), 28);
    assertEquals(tokensList[2].getList(0).size(), 29);
    assertEquals(tokensList[3].getList(0).size(), 23);
    assertEquals(tokensList[4].getList(0).size(), 20);
    assertEquals(tokensList[5].getList(0).size(), 0);
  }

  @Test
  public void testSaveAndLoad() throws IOException {
    KuromojiTokenizer tokenizer = new KuromojiTokenizer()
        .setInputCol("text")
        .setOutputCol("tokens")
        .setMode("EXTENDED");
    String path = File.createTempFile("spark-kuromoji-tokenizer", "java").getAbsolutePath();
    tokenizer.write().overwrite().save(path);
    KuromojiTokenizer loaded = KuromojiTokenizer.load(path);
    assertEquals(loaded.getMode(), "EXTENDED");
  }
}
