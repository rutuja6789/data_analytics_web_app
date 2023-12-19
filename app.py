from flask import Flask, render_template, request, jsonify
from matplotlib_inline.backend_inline import FigureCanvas
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, desc
from pyspark.sql.types import IntegerType
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

app = Flask(__name__)

# Create a Spark session
spark = SparkSession.builder.master("local[*]").appName("data_analytics_web_app").getOrCreate()

# Path to your CSV file
csv_path = "flights_sample_3m.csv"
print(f"Attempting to read CSV from: {csv_path}")

# Read the CSV file into a PySpark DataFrame
df = spark.read.format("csv").option("header", "true").load(csv_path)

# Inside generate_plot function in your Flask app
def generate_plot(spark_df, selected_year):
    print(f"Generating plot for the year {selected_year}...")

    try:
        plt.figure(figsize=(10, 6))

        # Extract the year from the "FL_DATE" column using PySpark functions
        spark_df = spark_df.withColumn("YEAR", year(col("FL_DATE")))

        # Filter the data based on the selected year
        df_filtered_year = spark_df.filter(col("YEAR") == int(selected_year))

        # Convert "DEP_DELAY" to numeric
        df_filtered_year = df_filtered_year.withColumn("DEP_DELAY", col("DEP_DELAY").cast(IntegerType()))

        # Convert the PySpark DataFrame to a Pandas DataFrame
        df_pandas = df_filtered_year.toPandas()

        # Calculate average delay per day
        average_daily_delay = df_pandas.groupby(df_pandas["FL_DATE"])["DEP_DELAY"].mean().reset_index()

        # Plotting the average delay per day
        plt.scatter(average_daily_delay["FL_DATE"], average_daily_delay["DEP_DELAY"],color="blue")

        # Add title, labels, and legend
        plt.title(f"Average Departure Delay per Day - Year {selected_year}")
        plt.xlabel(f"Year {selected_year} ")
        plt.ylabel("Average Departure Delay")
        plt.legend(["Departure Delay"])

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format="png")
        image_stream.seek(0)

        # Encode the plot image as base64 for HTML rendering
        plot_image = base64.b64encode(image_stream.read()).decode("utf-8")

        # Print the statement that the code is generated
        print("Code is generated!")

        # Replace placeholder image with the generated image
        return f"data:image/png;base64,{plot_image}"

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception to see the full traceback


@app.route("/generateVisual", methods=["GET", "POST"])
def dashboard():
    print("Accessing the index route...")

    print(request.method)

    if request.method == "POST":
        selected_year = request.form.get("year")
        # Generate data visualization plot based on the selected year
        plot_image = generate_plot(df, selected_year)
    else:
        # Default: Generate data visualization plot for the first year in the dropdown
        return render_template("visual.html")

    # Render the HTML template and pass the plot image as an argument
    return render_template("visual.html", plot_image=plot_image)

@app.route('/dataset_details', methods=['GET'])
def get_dataset_details():
    total_count = df.count()
    sample_data = df.limit(5).toPandas().to_dict(orient='records')
    mean_delay = df.select(col("DEP_DELAY").cast("double")).agg({"DEP_DELAY": "mean"}).collect()[0][0]
    data_desc = df.describe().toPandas().to_dict(orient='records')
    print(data_desc)
    return render_template('dataset.html', total_count=total_count,mean_delay=mean_delay, data_desc=data_desc, sample_data=sample_data)
print(df.head(2))

def generate_airline_plot(spark_df):
    print("Generating plot for the number of flights per airline...")
    spark_df = spark_df.withColumn("YEAR", year(col("FL_DATE")))

    try:
        plt.figure(figsize=(12, 6))

        # Group by 'AIRLINE_CODE' and count the number of repetitions
        airline_counts = spark_df.groupBy('AIRLINE_CODE').count().toPandas()

        # Plotting the counts
        plt.bar(airline_counts['AIRLINE_CODE'], airline_counts['count'], color='skyblue')

        # Add title, labels, and legend
        plt.title('Number of Flights per Airline')
        plt.xlabel('Airline Names')
        plt.ylabel('Number of Flights')
        plt.xticks(rotation=45, ha='right')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format="png")
        image_stream.seek(0)

        # Encode the plot image as base64 for HTML rendering
        plot_image = base64.b64encode(image_stream.read()).decode("utf-8")

        # Print the statement that the code is generated!
        print("Code is generated!")

        return plot_image

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception to see the full traceback

def generate_destination_chart(spark_df):
    print("Generating plot for the most popular destination...")

    try:
        plt.figure(figsize=(12, 6))

        # Group by 'DEST_CITY' and count the number of repetitions, then sort in descending order
        destination_counts = spark_df.groupBy('DEST').count().orderBy(desc('count')).toPandas()

        # Plotting the counts
        plt.bar(destination_counts['DEST'], destination_counts['count'], color='skyblue')

        # Add title, labels, and legend
        plt.title('Most Popular Destination Cities')
        plt.xlabel('Destination Cities')
        plt.ylabel('Number of Flights')
        plt.xticks(rotation=45, ha='right')

        # Save the plot to a BytesIO object
        image_stream = BytesIO()
        plt.savefig(image_stream, format="png")
        image_stream.seek(0)

        # Encode the plot image as base64 for HTML rendering
        plot_image = base64.b64encode(image_stream.read()).decode("utf-8")

        # Print the statement that the code is generated!
        print("Destination chart is generated!")

        return plot_image

    except Exception as e:
        print(f"An error occurred: {e}")
        raise  # Re-raise the exception to see the full traceback

@app.route('/generate_airline_plot', methods=['GET'])
def generate_airline_plot_route():
    plot_image = generate_airline_plot(df)
    plot_image_2=generate_destination_chart(df)
    return render_template("dataset_explore.html", plot_image=plot_image, plot_image_2=plot_image_2)

@app.route("/")
def index():

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='8080')
