import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


# read dl.cfg file that contains AWS credentials
config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


# create spark session with hadoop-aws package
def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    
    """
    This function reads song_data dataset from S3, processes data using Spark and writes them back to S3 in form of songs_table and artists_table parquet files.
        
    Input parameters:
            spark: Spark Session
            input_data  : string with song_data S3 location
            output_data : string with S3 location where created tables should be stored
    """
    
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data/A/A/A/*.json")
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table (table 1)
    songs_table = df['song_id', 'title', 'artist_id', 'year', 'duration']
    songs_table = songs_table.dropDuplicates(['song_id'])
    
    # write songs table to parquet files partitioned by song's year and artist's id
    songs_table.write \
               .partitionBy('year','artist_id') \
               .parquet(os.path.join(output_data,'songs.parquet'),'overwrite')

    # extract columns to create artists table (table 2)
    artists_table = df['artist_id','artist_name','artist_location','artist_latitude','artist_longitude']
    
    # write artists table to parquet files
    artists_table.write \
                 .parquet(os.path.join(output_data,'artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    
    """
        This function reads log_data dataset from S3, processes data using Spark and writes them back to S3 in form of songplays_table, users_table and time_table parquet files.
        
        Parameters:
            spark: Spark Session
            input_data  : string with log_data S3 location
            output_data : string with S3 location where created tables should be stored
            
    """
    
    # get filepath to log data file
    log_data = os.path.join(input_data,"log_data/*/*/*.json")
    
    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays (table 3)
    songplays_table = df['ts','userId','level','sessionId','location','userAgent']

    # extract columns for users table (table 4)
    users_table = df['userId','firstName','lastName','gender','level']
    users_table = users_table.dropDuplicates(['userId'])
    
    # write users table to parquet files
    users_table.write \
               .parquet(os.path.join(output_data, 'users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    df = df.withColumn('datetime', get_datetime(df.ts))
    
    
    # extract columns to create time table (table 5)
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'),
        year('datetime').alias('year') 
   )
    
    time_table = time_table.dropDuplicates(['start_time'])
    
    
    # write time table to parquet files partitioned by year and month
    time_table.write \
              .partitionBy('year', 'month') \
              .parquet(os.path.join(output_data, 'time.parquet'), 'overwrite')

    # read in song data to use for songplays table
    song_data = os.path.join(input_data, "song_data/A/A/A/*.json")
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
      
    df = df[df.page == 'NextSong']
    
    songplays_table = df.join(song_df, (df.song == song_df.title) & (df.artist == song_df.artist_name) & (df.length == song_df.duration), 'left_outer')\
        .select(
            df.timestamp,
            col("userId").alias('user_id'),
            df.level,
            song_df.song_id,
            song_df.artist_id,
            col("sessionId").alias("session_id"),
            df.location,
            col("useragent").alias("user_agent"),
            year('datetime').alias('year'),
            month('datetime').alias('month')
        )
    

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
                   .partitionBy('year', 'month') \
                   .parquet(os.path.join(output_data, 'songplays.parquet'), 'overwrite')
    
    
def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend-lysmar/"
    output_data = "s3a://data-lake-spark-lysmar/"
    
    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)
    
if __name__ == "__main__":
    main()


