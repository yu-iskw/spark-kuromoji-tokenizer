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

import scala.collection.JavaConverters._
import org.atilika.kuromoji.{Token => KToken, Tokenizer => KTokenizer}
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * :: Experimental ::
  * A Spark transformer class to tokenize with Kuromoji
  *
  * Kuromoji is an open source Japanese morphological analyzer written in Java.
  */
@Experimental
class KuromojiTokenizer(override val uid: String)
  extends Transformer with DefaultParamsWritable with KuromojiTokenizerParams {

  // Sets the default values
  setDefault(
    mode -> "NORMAL",
    dictPath -> KuromojiTokenizer.DICT_PATH_NULL_VALUE
  )

  def this() = this(Identifiable.randomUID("kuromojitok"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group expertSetParam */
  def setMode(value: String): this.type = set(mode, value)

  /** @group expertSetParam */
  def setDictPath(value: String): this.type = set(dictPath, value)

  override def copy(extra: ParamMap): KuromojiTokenizer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    val inputType = schema($(inputCol)).dataType
    validateInputType(inputType)
    if (schema.fieldNames.contains($(outputCol))) {
      throw new IllegalArgumentException(s"Output column ${$(outputCol)} already exists.")
    }
    val outputFields = schema.fields :+
      StructField($(outputCol), new ArrayType(StringType, true), nullable = false)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    val tokenUDF = udf { text: String =>
      val modeClass = KuromojiTokenizer.addressMode($(mode))
      $(dictPath) match {
        case KuromojiTokenizer.DICT_PATH_NULL_VALUE =>
          CustomKuromojiTokenizer.tokenize(text, modeClass).map(_.getSurfaceForm)
        case _ =>
          CustomKuromojiTokenizer.tokenize(text, modeClass, $(dictPath)).map(_.getSurfaceForm)
      }
    }

    transformSchema(dataset.schema, logging = true)
    dataset.withColumn($(outputCol),tokenUDF( dataset($(inputCol))))

  }

  /*
  protected def createTransformFunc: String => Seq[String] = {
    val modeClass = KuromojiTokenizer.addressMode($(mode))
    $(dictPath) match {
      case KuromojiTokenizer.DICT_PATH_NULL_VALUE =>
        CustomKuromojiTokenizer.tokenize(_, modeClass).map(_.getSurfaceForm)
      case _ =>
        CustomKuromojiTokenizer.tokenize(_, modeClass, $(dictPath)).map(_.getSurfaceForm)
    }
  }
  */

  protected def outputDataType: DataType = new ArrayType(StringType, true)

  protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }
}


/**
  * :: Experimental ::
  * A Spark object to deal with Kuromoji tokenizer
  */
object KuromojiTokenizer extends DefaultParamsReadable[KuromojiTokenizer] {

  private[feature] val DICT_PATH_NULL_VALUE = "NULL_VALUE"

  override def load(path: String): KuromojiTokenizer = super.load(path)

  private[feature] def addressMode(mode: String): KTokenizer.Mode = {
    mode match {
      case "NORMAL" => KTokenizer.Mode.NORMAL
      case "SEARCH" => KTokenizer.Mode.SEARCH
      case "EXTENDED" => KTokenizer.Mode.EXTENDED
      case _ => throw new IllegalArgumentException(s"${mode} is invalid. " +
        s"You should go with NORMAL, SEARCH or EXTENDED")
    }
  }
}


/**
  * :: Experimental ::
  * Parameter trait for `KuromojiTokenizer`
  */
private[feature]
trait KuromojiTokenizerParams extends Params with HasInputCol with HasOutputCol {

  /**
    * Set the Kuromoji mode
    * @group expertParam
    */
  final val mode = new Param[String](this, "mode", "mode")

  /** @group expertGetParam */
  def getMode: String = $(mode)

  /**
    * Set the path to the dictionary for Kuromoji
    * @group expertParam
    */
  final val dictPath = new Param[String](this, "dictPath", "path to a dictionary")

  /** @group expertGetParam */
  def getDictionaryPath: String = $(dictPath)
}


/**
  * An object to tokenize with Kuromoji
  * TODO We can make it more efficient. Because this version build a tokenizer each times.
  */
private[feature]
object CustomKuromojiTokenizer {

  def tokenize(text: String, mode: KTokenizer.Mode): Seq[KToken] = {
    val tokenizer = KTokenizer.builder().mode(mode).build()
    //tokenizer.tokenize(text).asScala.dropWhile(_.getSurfaceForm == ' ').toSeq
     tokenizer.tokenize(text).asScala.toSeq.filter(_.getSurfaceForm.trim().nonEmpty)
  
  }

  def tokenize(text: String, mode: KTokenizer.Mode, dictPath: String): Seq[KToken] = {
    val tokenizer = KTokenizer.builder().mode(mode).userDictionary(dictPath).build()
    tokenizer.tokenize(text).asScala.toSeq.filter(_.getSurfaceForm.trim().nonEmpty)
    //tokenizer.tokenize(text).asScala.toSeq
  }
}
