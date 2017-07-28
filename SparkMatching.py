from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import *

import math

import os

#import numpy as np
#
# def ngramTokenizer(data,ngram):
#     mergedData = ''.join(data).replace(" ",'')
#     return [mergedData[i:i + ngram] for i in range(len(mergedData) - (ngram - 1))]


#reads a list of strings
# output one big list of words combined
#ex: ['GIRL FROM IPANMEMA','HELLIS HORBARD','COMP SOUNDER']
#ourput:['GIRL','FROM','IPNEMA','HELLIS','HORBARD','COMP','SOUNDER']

def ngramTokenizer(data,ngram):
    mergedData = ' '.join(data)
    return mergedData.split(' ')

# Each word in a list is sent out as a record
# ex: [1,'GIRL FROM IPANEMA']
#[(1,'GIRL',0.33),(1,'FROM',0.33),(1,'IPANEMA',0.33)]

def tf(index,data):
    dataset= set(data)
    rowlength =len(dataset)
    rl=[]

    for i in dataset:
        rl.append((index,i,float(data.count(i))/rowlength))

    return rl

def main(reffile,sourcefile):

    sc = SparkContext()

    spark = SparkSession(sc)

    if os.path.exists('/Users/sharath/Datasets/refidf_parquet/refidf.parquet') and os.path.exists('/Users/sharath/Datasets/reftfidf_parquet/reftidf.parquet'):

        reftfidf=spark.read.load('/Users/sharath/Datasets/reftfidf_parquet/reftidf.parquet')
        ref_idf = spark.read.load('/Users/sharath/Datasets/refidf_parquet/refidf.parquet')
    else:
        reftfidf,ref_idf =build_tf_idf(sc,spark,reffile)



    srctfidf = build_src_tf_idf(sc,spark,sourcefile,ref_idf)

    src_tfidf = srctfidf.select(srctfidf.id.alias("src_id"),srctfidf.word,srctfidf.tfidf.alias("src_tfidf"))

    #print reftfidf.show(5)

    #print src_tfidf.show()
    matching(spark,reftfidf,src_tfidf)


def matching(spark,reftfidf,srctfidf):

    ref_src_df=reftfidf.join(srctfidf,srctfidf.word==reftfidf.word)

    ref_src_df.createOrReplaceTempView('ref_src')

    #print ref_src_df.show(5)

    matched_df = spark.sql('select src_id,id,sum(tfidf*src_tfidf) score from ref_src group by id,src_id having sum(tfidf*src_tfidf)>0.8')

    matched_df.show()



def build_tf_idf(sc,spark,ipfile):

    print 'begin building ref index'

    twa_rdd = sc.textFile(ipfile).map(lambda x: (x.split('|')[0],x.split('|')[1:]))

    total_docs = twa_rdd.count()

    print 'total number of lines',total_docs

    ngram_twa_rdd = twa_rdd.map(lambda x:(x[0],ngramTokenizer(x[1],3)))

    ref_twa_tf = ngram_twa_rdd.map(lambda x: tf(x[0],x[1]))

    ref_twa_flat_map = ref_twa_tf.flatMap(lambda x: x)

    total_ref_tf = ref_twa_flat_map.count()


    ref_tf_df = spark.createDataFrame(ref_twa_flat_map,['id','word','tf'])



    ref_tf_df.createOrReplaceTempView('reftf')

    query = 'select word, ({}/count(distinct id)) idf from reftf group by word'.format(total_docs)


    pre_idf = spark.sql(query)


    ref_idf = pre_idf.select("word",log(pre_idf.idf).alias("idf"))


    pre_tf_idf = ref_tf_df.join(ref_idf,ref_tf_df.word==ref_idf.word)



    tf_idf = pre_tf_idf.select(pre_tf_idf.id,ref_tf_df.word,(pre_tf_idf.tf*pre_tf_idf.idf).alias("tfidf"))

    tf_idf.createOrReplaceTempView('reftfidf')

    tf_idf_len = spark.sql('select id, sqrt(sum(tfidf*tfidf)) length from reftfidf group by id')


    pre_norm_tfidf = tf_idf.join(tf_idf_len,tf_idf.id==tf_idf_len.id)

    norm_tf_idf = pre_norm_tfidf.select(tf_idf.id,tf_idf.word,(tf_idf.tfidf/tf_idf_len.length).alias("tfidf"))

    norm_tf_idf.select("id","word","tfidf").write.save('/Users/sharath/Datasets/reftfidf_parquet/reftidf.parquet',format='parquet')

    ref_idf.select("word","idf").write.save("/Users/sharath/Datasets/refidf_parquet/refidf.parquet",format='parquet')


    return norm_tf_idf,ref_idf


def build_src_tf_idf(sc, spark, ipfile,ref_idf):

    print 'begin building source index'

    twa_rdd = sc.textFile(ipfile).map(lambda x: (x.split('|')[0], x.split('|')[1:]))

    total_docs = twa_rdd.count()

    print 'total number of source lines', total_docs

    ngram_twa_rdd = twa_rdd.map(lambda x: (x[0], ngramTokenizer(x[1], 3)))

    ref_twa_tf = ngram_twa_rdd.map(lambda x: tf(x[0], x[1]))

    ref_twa_flat_map = ref_twa_tf.flatMap(lambda x: x)

    total_ref_tf = ref_twa_flat_map.count()

    ref_tf_df = spark.createDataFrame(ref_twa_flat_map, ['id', 'word', 'tf'])


    pre_tf_idf = ref_tf_df.join(ref_idf, ref_tf_df.word == ref_idf.word)

    tf_idf = pre_tf_idf.select(pre_tf_idf.id, ref_tf_df.word, (pre_tf_idf.tf * pre_tf_idf.idf).alias("tfidf"))

    tf_idf.createOrReplaceTempView('reftfidf')

    tf_idf_len = spark.sql('select id, sqrt(sum(tfidf*tfidf)) length from reftfidf group by id')

    pre_norm_tfidf = tf_idf.join(tf_idf_len, tf_idf.id == tf_idf_len.id)

    norm_tf_idf = pre_norm_tfidf.select(tf_idf.id, tf_idf.word, (tf_idf.tfidf / tf_idf_len.length).alias("tfidf"))

    return norm_tf_idf


if __name__=='__main__':
    print 'in main'
    main('/Users/sharath/Datasets/Indexed_TWA.txt','/Users/sharath/Datasets/src_indexed_twa.txt')
