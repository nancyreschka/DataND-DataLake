import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, monotonically_increasing_id
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType, DateType, LongType, TimestampType


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    - Creates a spark session with spark and hadoop.
    
    Parameters:
        None

    Returns:
        spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark

def read_song_data(spark, input_data):
    """
    - Reads the song_data from input_data path into a spark dataframe
    
    Parameters:
        spark: spark session
        input_data: path from where the data should be read

    Returns:
        song_data dataframe
    """
    # get filepath to song data file
    song_data_path = f'{input_data}/song_data/*/*/*/*.json'
    
    # define schema for data to be read in
    songdataSchema = StructType([
        StructField("num_songs", IntegerType()),
        StructField("artist_id", StringType()),
        StructField("artist_latitude", DoubleType()),
        StructField("artist_longitude", DoubleType()),
        StructField("artist_location", StringType()),
        StructField("artist_name", StringType()),
        StructField("song_id", StringType()),
        StructField("title", StringType()),
        StructField("duration", DoubleType()),
        StructField("year", IntegerType()),
    ])
    
    # read song data file
    song_df = spark.read.json(song_data_path, songdataSchema)
    return song_df

def process_song_data(spark, input_data, output_data):
    """
    - Reads the song_data from input_data path into a spark dataframe
    - Extracts the necessary data for songs and artists table
    - Stores the analytical tables in output_data path in parquet files
    
    Parameters:
        spark: spark session
        input_data: path from where the data should be read
        output_data: path where the data should be written

    Returns:
        None
    """
    # read song data file
    song_df = read_song_data(spark, input_data)
    song_df.createOrReplaceTempView("song_data")

    # extract columns to create songs table
    songs_table = spark.sql("""
        SELECT song_id, title, artist_id, year, duration
        FROM song_data
    """)
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(os.path.join(output_data, "songs/"), mode='overwrite', partitionBy=["year","artist_id"])

    # extract columns to create artists table
    artists_table = spark.sql("""
        SELECT artist_id, artist_name AS name, artist_location AS location, artist_latitude AS latitude, 
        artist_longitude AS longitude
        FROM song_data
    """)
    
    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, "artists/"), mode='overwrite')


def process_log_data(spark, input_data, output_data):
    """
    - Reads the log_data from input_data path into a spark dataframe
    - Extracts the necessary data for users, time, and songplays table
    - Stores the analytical tables in output_data path in parquet files
    
    Parameters:
        spark: spark session
        input_data: path from where the data should be read
        output_data: path where the data should be written

    Returns:
        None
    """
    # get filepath to log data file
    log_data_path = f'{input_data}/log_data/'
    
    # define schema for data to be read in
    logdataSchema = StructType([
        StructField("artist", StringType()),
        StructField("auth", StringType()),
        StructField("firstName", StringType()),
        StructField("gender", StringType()),
        StructField("itemInSession", IntegerType()),
        StructField("lastName", StringType()),
        StructField("length", DoubleType()),
        StructField("level", StringType()),
        StructField("location", StringType()),
        StructField("method", StringType()),
        StructField("page", StringType()),
        StructField("registration", DoubleType()),
        StructField("sessionId", IntegerType()),
        StructField("song", StringType()),
        StructField("status", IntegerType()),
        StructField("ts", LongType()),
        StructField("userAgent", StringType()),
        StructField("userId", StringType()),
    ])

    # read log data file
    log_df = spark.read.json(log_data_path, logdataSchema)
    log_df = log_df.where("page == 'NextSong'")
    
    # filter by actions for song plays
    log_df.createOrReplaceTempView("log_data")

    # extract columns for users table    
    users_table = spark.sql("""
        SELECT DISTINCT s1.userId AS user_id, 
            s1.firstName AS first_name, 
            s1.lastName AS last_name, 
            s1.gender, 
            s1.level
        FROM log_data s1
        WHERE s1.userId IS NOT NULL 
        AND s1.ts = (SELECT max(s2.ts)
                            FROM log_data s2
                            WHERE s1.userId = s2.userId)    
    """)
    
    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, "users/"), mode='overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000.0), TimestampType())
    log_df = log_df.withColumn('start_time', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x / 1000.0).strftime('%Y-%m-%d %H:%M:%S'), StringType())
    log_df = log_df.withColumn('date_time', get_datetime('ts'))
    
    # extract columns to create time table
    log_df.createOrReplaceTempView("log_data")
    time_table = spark.sql("""
        SELECT date_time AS start_time,
            hour(start_time) AS hour, 
            day(start_time) AS day, 
            weekofyear(start_time) AS week, 
            month(start_time) AS month, 
            year(start_time) AS year, 
            weekday(start_time) AS weekday 
        FROM log_data
    """)
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(os.path.join(output_data, "time/"), mode='overwrite', partitionBy=["year","month"])
    
    # read in song data to use for songplays table
    song_df = read_song_data(spark, input_data)
    
    # join the songs with the log data
    songplays_df = song_df.join(log_df, (song_df.title == log_df.song) & (song_df.artist_name == log_df.artist) & ((song_df.duration == log_df.length)))
    
    # add the id column to the dataframe
    songplays_df = songplays_df.withColumn('songplay_id', monotonically_increasing_id())
    songplays_df.createOrReplaceTempView("songplays")

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""
        SELECT 
            songplay_id,
            date_time AS start_time, 
            userId AS user_id, 
            level, 
            song_id, 
            artist_id, 
            sessionId AS session_id, 
            location, 
            userAgent AS user_agent,
            month(start_time) AS month, 
            year(start_time) AS year
        FROM songplays
        WHERE userId IS NOT NULL
            AND level IS NOT NULL
            AND sessionId IS NOT NULL
            AND location IS NOT NULL
            AND userAgent IS NOT NULL                                
            AND date_time IS NOT NULL
            AND song_id IS NOT NULL
            AND artist_id IS NOT NULL
    """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(os.path.join(output_data, "songplays/"), mode='overwrite', partitionBy=["year","month"])


def main():
    """
    - Creates spark session
    - Processes the song_data
    - Processes the log_data
    
    Parameters:
        None

    Returns:
        None
    """
    spark = create_spark_session()
    input_data = config['PATHS']['INPUT_PATH']
    output_data = config['PATHS']['OUTPUT_PATH']
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
