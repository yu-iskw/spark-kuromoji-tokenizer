# Kuromoji Tokenizer for Spark DataFrames

[![Build Status](https://travis-ci.org/yu-iskw/spark-kuromoji-tokenizer.svg?branch=master)](https://travis-ci.org/yu-iskw/spark-kuromoji-tokenizer)
[![codecov](https://codecov.io/gh/yu-iskw/spark-kuromoji-tokenizer/branch/master/graph/badge.svg)](https://codecov.io/gh/yu-iskw/spark-kuromoji-tokenizer)

This is a Kuromoji tokenizer as a `Transformer` on Spark DataFrame.

## Requirement

- Apache Spark: 1.6 or higher

## Example

We can use this package like bellow:

```scala
// Creates a sample data which has a column whose name is `text`.
// The column involves Japanese text.
val data = Seq(
  Row("天皇は、日本国の象徴であり日本国民統合の象徴であつて、この地位は、主権の存する日本国民の総意に基く。"),
  Row("皇位は、世襲のものであつて、国会の議決した皇室典範 の定めるところにより、これを継承する。"),
  Row("天皇の国事に関するすべての行為には、内閣の助言と承認を必要とし、内閣が、その責任を負ふ。"),
  Row("天皇は、この憲法の定める国事に関する行為のみを行ひ、国政に関する権能を有しない。"),
  Row("天皇は、法律の定めるところにより、その国事に関する行為を委任することができる。")
)
val schema = StructType(Seq(StructField("text", StringType, false)))
val df = sqlContext.createDataFrame(sc.parallelize(data), schema)

// Tokenizes with this package.
import org.atilika.kuromoji.{Tokenizer => KTokenizer}
val kuromoji = new KuromojiTokenizer()
  .setInputCol("text")
  .setOutputCol("tokens")
  .setMode(KTokenizer.Mode.NORMAL) // Optional
  .setDictPath(pathToDictionary)   // Optional
val transformed = kuromoji.transform(df)
```

`transformed` is much kind of like the following.

|text                                              |tokens                                                                                                            |
|:--------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|
|天皇は、日本国の象徴であり日本国民統合の象徴であつて、この地位は、主権の存する日本国民の総意に基く。|[天皇, は, 、, 日本, 国, の, 象徴, で, あり, 日本, 国民, 統合, の, 象徴, で, あ, つて, 、, この, 地位, は, 、, 主権, の, 存する, 日本, 国民, の, 総意, に, 基く, 。]|
|皇位は、世襲のものであつて、国会の議決した皇室典範 の定めるところにより、これを継承する。     |[皇位, は, 、, 世襲, の, もの, で, あ, つて, 、, 国会, の, 議決, し, た, 皇室, 典範,  , の, 定める, ところ, により, 、, これ, を, 継承, する, 。]             |
|天皇の国事に関するすべての行為には、内閣の助言と承認を必要とし、内閣が、その責任を負ふ。      |[天皇, の, 国事, に関する, すべて, の, 行為, に, は, 、, 内閣, の, 助言, と, 承認, を, 必要, と, し, 、, 内閣, が, 、, その, 責任, を, 負, ふ, 。]            |
|天皇は、この憲法の定める国事に関する行為のみを行ひ、国政に関する権能を有しない。          |[天皇, は, 、, この, 憲法, の, 定める, 国事, に関する, 行為, のみ, を, 行, ひ, 、, 国政, に関する, 権能, を, 有, し, ない, 。]                            |
|天皇は、法律の定めるところにより、その国事に関する行為を委任することができる。           |[天皇, は, 、, 法律, の, 定める, ところ, により, 、, その, 国事, に関する, 行為, を, 委任, する, こと, が, できる, 。]                                   |


## Parameters

### Required Parameters

- `setInputCol`: Input column name. 
- `setOutputCol`: Output column name.

### Optional Expert Parameters
- `setMode`: Kuromoji mode. Default value is `org.atilika.kuromoji.Tokenizer.Mode.Normal`.
- `setDictPath`: Path to dictionary path.
