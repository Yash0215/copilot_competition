import os
import logging
from pyspark.sql.window import Window
from pyspark.sql.functions import col
from pyspark.sql.functions import udf
from math import radians, sin, cos, sqrt, atan2
from pyspark.sql.functions import first
from pyspark.sql import functions as F
from pyspark.sql.functions import concat, concat_ws

# PNR_DATA_ADLS_PATH
dbutils.widgets.text("PNR_DATA_ADLS_PATH", "/mnt/ppeedp/raw/competition/pnr_sample")

# DISTANCE_MASTER_ADLS_PATH
dbutils.widgets.text("DISTANCE_MASTER_ADLS_PATH", "/mnt/stppeedp/ppeedp/CBI2/production/reference_zone/distance_master")

# LOCATION_MASTER_ADLS_PATH 
dbutils.widgets.text("LOCATION_MASTER_ADLS_PATH", "/mnt/stppeedp/ppeedp/raw/eag/ey/test_cbi_reference_data_loader/target_dir/Location_master")

# BACKTRACK_MASTER_ADLS_PATH
dbutils.widgets.text("BACKTRACK_MASTER_ADLS_PATH", "/mnt/stppeedp/ppeedp/copilot/eyychawda/BacktrackExceptionmaster")

# FINAL_OUTPUT_PATH
dbutils.widgets.text("FINAL_OUTPUT_PATH", "/mnt/stppeedp/ppeedp/copilot/eyychawda/output")


pnr_data_adls_path = dbutils.widgets.get("PNR_DATA_ADLS_PATH")
distance_master_adls_path = dbutils.widgets.get("DISTANCE_MASTER_ADLS_PATH")
location_master_adls_path = dbutils.widgets.get("LOCATION_MASTER_ADLS_PATH")
backtrack_master_adls_path = dbutils.widgets.get("BACKTRACK_MASTER_ADLS_PATH")
final_output_path = dbutils.widgets.get("FINAL_OUTPUT_PATH")

# Get the basename of the current Databricks Notebook
notebook_name = os.path.basename(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

# Create logger with notebook name
logger = logging.getLogger(notebook_name)
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')

# Create console handler and set level and formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Test the logger
logger.info("This is a test log message")

# Log the name of all five input parameters
logger.info("PNR_DATA_ADLS_PATH: %s", pnr_data_adls_path)
logger.info("DISTANCE_MASTER_ADLS_PATH: %s", distance_master_adls_path)
logger.info("LOCATION_MASTER_ADLS_PATH: %s", location_master_adls_path)
logger.info("BACKTRACK_MASTER_ADLS_PATH: %s", backtrack_master_adls_path)
logger.info("FINAL_OUTPUT_PATH: %s", final_output_path)

try:
    # Read PNR_DATA_ADLS_PATH as delta format
    pnr_sample = spark.read.format("delta").load(pnr_data_adls_path)
    pnr_sample.createOrReplaceTempView("pnr_sample")
    logger.info("pnr_sample count: %d", pnr_sample.count())

    # Read DISTANCE_MASTER_ADLS_PATH as delta format
    distance_master = spark.read.format("delta").load(distance_master_adls_path)
    distance_master.createOrReplaceTempView("distance_master")
    logger.info("distance_master count: %d", distance_master.count())

    # Read LOCATION_MASTER_ADLS_PATH as delta format
    location_master = spark.read.format("delta").load(location_master_adls_path)
    location_master.createOrReplaceTempView("location_master")
    logger.info("location_master count: %d", location_master.count())

    # Read BACKTRACK_MASTER_ADLS_PATH as delta format
    backtrack_master = spark.read.format("delta").load(backtrack_master_adls_path)
    backtrack_master.createOrReplaceTempView("backtrack_master")
    logger.info("backtrack_master count: %d", backtrack_master.count())

except Exception as e:
    logger.error("An exception occurred: %s", str(e))
    dbutils.notebook.exit("Notebook execution failed due to an exception")


# Create window spec for OD Origin
pnr_asc = Window.partitionBy(col("PNRHash")).orderBy(col("SEG_SEQ_NBR").asc())

# Create window spec for OD Destination
pnr_desc = Window.partitionBy(col("PNRHash")).orderBy(col("SEG_SEQ_NBR").desc())

def dms_to_decimal(dms):
    # Remove the alphabet character at the end if present
    dms = dms.rstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # Split the dms string into degree, minutes, and seconds
    degree, minutes, seconds = dms.split('.')
    
    # Calculate the decimal degrees
    decimal_degrees = float(degree) + float(minutes)/60 + float(seconds)/3600
    
    return decimal_degrees


# UDF to calculate distance using Haversine formula
@udf
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on the Earth's surface using Haversine formula.
    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.
    Returns:
        float: Distance between the two points in kilometers.
    """

    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Calculate the differences in latitude and longitude
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Calculate the Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance

# Register the UDF with Spark
spark.udf.register("calculate_distance", calculate_distance)


pnr_with_od = pnr_sample.withColumn("od_orig", first(col("orig_ap_cd")).over(pnr_asc)) \
                        .withColumn("od_dest", first(col("destn_ap_cd")).over(pnr_desc)) \
                        .select("*")

pnr_with_od.createOrReplaceTempView("pnr_with_od")

# Perform LEFT JOIN between pnr_with_od and backtrack_master
pnr_with_breakmaster = spark.sql("""
    SELECT p.*, 
           SUM(CASE WHEN b.orig IS NOT NULL AND b.dest IS NOT NULL THEN 1 ELSE 0 END) OVER (PARTITION BY p.PNRHash) AS is_break_sum
    FROM pnr_with_od p
    LEFT JOIN backtrack_master b
    ON p.orig_ap_cd = b.orig AND p.destn_ap_cd = b.dest
""")

# Create DataFrame and Temp View as "pnr_with_breakmaster"
pnr_with_breakmaster.createOrReplaceTempView("pnr_with_breakmaster")


# Perform LEFT join between pnr_with_breakmaster and distance_master
pnr_with_distance = spark.sql("""
    SELECT p.*, d1.distance_in_km AS segment_distance, d2.distance_in_km AS od_distance
    FROM pnr_with_breakmaster p
    LEFT JOIN distance_master d1
    ON p.orig_ap_cd = d1.airport_origin_code AND p.destn_ap_cd = d1.airport_destination_code
    LEFT JOIN distance_master d2
    ON p.od_orig = d2.airport_origin_code AND p.od_dest = d2.airport_destination_code
""")

# Create DataFrame and Temp View as "pnr_with_distance"
pnr_with_distance.createOrReplaceTempView("pnr_with_distance")


# 1. import spark sql functions as F

# 2. Get count of rows where "segment_distance" is NULL from "pnr_with_distance"
null_segment_distance_count = pnr_with_distance.filter(F.col("segment_distance").isNull()).count()

# 3. Log the "null_segment_distance_count"
logger.info("null_segment_distance_count: %d", null_segment_distance_count)

# 4. Do following if null_segment_distance_count is greater than 0:
if null_segment_distance_count > 0:
    # 4.1 Apply the dms to decimal UDF to "location_master" on "latitude" and "longitude" respectively by using F.expr
    location_master = location_master.withColumn("latitude_decimal", F.expr("dms_to_decimal(latitude)"))
    location_master = location_master.withColumn("longitude_decimal", F.expr("dms_to_decimal(longitude)"))

    # 4.2 LEFT JOIN "pnr_with_distance" with "location_master" using left_table.orig_ap_cd=right_table.ap_cd_val and select right_table.latitude_decimal as origin_latitude_decimal and right_table.longitude_decimal as origin_longitude_decimal
    pnr_with_distance = pnr_with_distance.join(location_master, pnr_with_distance["orig_ap_cd"] == location_master["ap_cd_val"], "left") \
                                         .select(pnr_with_distance["*"], location_master["latitude_decimal"].alias("origin_latitude_decimal"), location_master["longitude_decimal"].alias("origin_longitude_decimal"))

    # 4.3 LEFT JOIN "pnr_with_distance" with "location_master" using left_table.destn_ap_cd=right_table.ap_cd_val and select right_table.latitude_decimal as dest_latitude_decimal and right_table.longitude_decimal as dest_longitude_decimal
    pnr_with_distance = pnr_with_distance.join(location_master, pnr_with_distance["destn_ap_cd"] == location_master["ap_cd_val"], "left") \
                                         .select(pnr_with_distance["*"], location_master["latitude_decimal"].alias("dest_latitude_decimal"), location_master["longitude_decimal"].alias("dest_longitude_decimal"))

    # 4.4 Case when "segment_distance" is not null calculated "segment_distance" by Haversian Formulae UDF on previously generated columns columns by casting them to double or "segment_distance" otherwise by using F.expr
    pnr_with_distance = pnr_with_distance.withColumn("segment_distance", F.when(F.col("segment_distance").isNull(), 
                                                                                calculate_distance(F.col("origin_latitude_decimal").cast("double"), 
                                                                                                   F.col("origin_longitude_decimal").cast("double"), 
                                                                                                   F.col("dest_latitude_decimal").cast("double"), 
                                                                                                   F.col("dest_longitude_decimal").cast("double")))
                                                                         .otherwise(F.col("segment_distance")))


# 1. Get count of rows where "od_distance" is NULL from "pnr_with_distance"
null_od_distance_count = pnr_with_distance.filter(F.col("od_distance").isNull()).count()

# 2. Log the "null_od_distance_count"
logger.info("null_od_distance_count: %d", null_od_distance_count)

# 3. Do following if null_od_distance_count is greater than 0:
if null_od_distance_count > 0:
    # 3.1 Apply the dms to decimal UDF to "location_master" on columns "latitude" and "longitude" columns respectively and output the columns latitude_decimal2 and longitude_decimal2 respectively by using F.expr
    location_master = location_master.withColumn("latitude_decimal2", F.expr("dms_to_decimal(latitude)"))
    location_master = location_master.withColumn("longitude_decimal2", F.expr("dms_to_decimal(longitude)"))

    # 3.2 LEFT JOIN "pnr_with_distance" with "location_master" using left_table.od_orig=right_table.ap_cd_val and select right_table.latitude_decimal2 as od_origin_latitude_decimal and right_table.longitude_decimal2 as od_origin_longitude_decimal 
    pnr_with_distance = pnr_with_distance.join(location_master, pnr_with_distance["od_orig"] == location_master["ap_cd_val"], "left") \
                                            .select(pnr_with_distance["*"], location_master["latitude_decimal2"].alias("od_origin_latitude_decimal"), location_master["longitude_decimal2"].alias("od_origin_longitude_decimal"))

    # 3.3 LEFT JOIN "pnr_with_distance" with "location_master" using left_table.od_dest=right_table.ap_cd_val and select right_table.latitude_decimal2 as od_detination_latitude_decimal and right_table.longitude_decimal2 as od_destination_longitude_decimal 
    pnr_with_distance = pnr_with_distance.join(location_master, pnr_with_distance["od_dest"] == location_master["ap_cd_val"], "left") \
                                            .select(pnr_with_distance["*"], location_master["latitude_decimal2"].alias("od_destination_latitude_decimal"), location_master["longitude_decimal2"].alias("od_destination_longitude_decimal"))

    # 3.4 Case when "od_distance" is not null calculated "od_distance" by Haversian Formulae UDF on previously generated columns columns by casting them to double or "od_distance" otherwise by using F.expr
    pnr_with_distance = pnr_with_distance.withColumn("od_distance", F.when(F.col("od_distance").isNull(), 
                                                                            calculate_distance(F.col("od_origin_latitude_decimal").cast("double"), 
                                                                                                F.col("od_origin_longitude_decimal").cast("double"), 
                                                                                                F.col("od_destination_latitude_decimal").cast("double"), 
                                                                                                F.col("od_destination_longitude_decimal").cast("double")))
                                                                    .otherwise(F.col("od_distance")))



# 1. Calculate new column named as "total_segment_distance" by taking sum of segment_distance partitioned by PNRHash
# 2. Order it by PNRHash, SEG_SEQ_NBR in ascending order
final_pnr_distance = pnr_with_distance.withColumn("total_segment_distance", F.sum("segment_distance").over(Window.partitionBy("PNRHash").orderBy("PNRHash", "SEG_SEQ_NBR")))

# Create DataFrame and Temp View as "final_pnr_distance"
final_pnr_distance.createOrReplaceTempView("final_pnr_distance")


# Select with column
pnr_with_distance = pnr_with_distance.withColumn("ORIG_AP_CD", F.when((F.col("od_distance") > F.col("total_segment_distance")) | (F.col("is_break_sum") >= 1), F.col("od_orig")).otherwise(F.col("ORIG_AP_CD"))) \
                                   .withColumn("DESTN_AP_CD", F.when((F.col("od_distance") > F.col("total_segment_distance")) | (F.col("is_break_sum") >= 1), F.col("od_dest")).otherwise(F.col("DESTN_AP_CD")))

# Recreate the Temp View with the same name
pnr_with_distance.createOrReplaceTempView("pnr_with_distance")



# Create DataFrame with the desired columns using multiline Spark SQL
od_final = spark.sql("""
    SELECT CONCAT(PNRHash, '-', fl_dt, '-', SEG_SEQ_NBR, '-', OPT_ALN_CD) AS id,
           CAST(fl_dt AS timestamp) AS PNRCreateDate,
           OPT_ALN_CD AS IssueAirlineCode,
           PNRHash AS PNR,
           SEG_SEQ_NBR AS SegSequenceNumber,
           CONCAT(ORIG_AP_CD, '-', DESTN_AP_CD) AS true_od,
           CASE WHEN OPT_ALN_CD = 'EY' THEN CONCAT(ORIG_AP_CD, '-', DESTN_AP_CD) ELSE NULL END AS online_od,
           CONCAT(od_orig, '-', od_dest) AS online_od_itinerary,
           CASE WHEN OPT_ALN_CD = 'EY' THEN od_distance ELSE NULL END AS online_od_distance,
           CASE WHEN od_orig IS NOT NULL THEN od_orig ELSE ORIG_AP_CD END AS point_of_commencement,
           CASE WHEN od_dest IS NOT NULL THEN od_dest ELSE DESTN_AP_CD END AS point_of_finish,
           CASE WHEN od_distance > total_segment_distance OR is_break_sum >= 1 THEN od_distance ELSE segment_distance END AS true_od_distance,
           COLLECT_LIST(CONCAT(OPT_ALN_CD, ' ', DESTN_AP_CD)) OVER (PARTITION BY PNRHash) AS journey_itinerary,
           od_orig
    FROM final_pnr_distance
""")

# Create Temp View as "od_final"
od_final.createOrReplaceTempView("od_final")


# Select with column
od_final = od_final.withColumn("journey_itinerary_2", concat(od_final["od_orig"], concat_ws(" ", od_final["journey_itinerary"])))

# Recreate the Temp View with the same name
od_final.createOrReplaceTempView("od_final")


PNR_OD_OUTPUT = spark.sql("""
    SELECT id,
           PNRCreateDate,
           IssueAirlineCode,
           PNR,
           SegSequenceNumber,
           true_od,
           online_od,
           online_od_itinerary,
           online_od_distance,
           concat_ws(" ", collect_list(journey_itinerary_2)) as journey_operating,
           point_of_commencement,
           point_of_finish,
           true_od_distance
    FROM od_final
""")

display(PNR_OD_OUTPUT)

# Log the counts of PNR_OD_OUTPUT
logger.info("PNR_OD_OUTPUT count: %d", PNR_OD_OUTPUT.count())

# Write the PNR_OD_OUTPUT DataFrame as a Delta table
PNR_OD_OUTPUT.write.format("delta").mode("overwrite").save(final_output_path + "/PNR_OD_OUTPUT")

logger.info("Code execution completed successfully.")
dbutils.notebook.exit("Code execution completed successfully.")