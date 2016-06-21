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
    List<String> data = Arrays.asList(
        "天皇は、日本国の象徴であり日本国民統合の象徴であつて、この地位は、主権の存する日本国民の総意に基く。",
        "皇位は、世襲のものであつて、国会の議決した皇室典範 の定めるところにより、これを継承する。",
        "天皇の国事に関するすべての行為には、内閣の助言と承認を必要とし、内閣が、その責任を負ふ。",
        "天皇は、この憲法の定める国事に関する行為のみを行ひ、国政に関する権能を有しない。",
        "天皇は、法律の定めるところにより、その国事に関する行為を委任することができる。");
    JavaRDD<Row> rdd = jsc.parallelize(data).map((String text) -> {
      return RowFactory.create(text);
    });
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
