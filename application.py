from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_data", methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            writing_score=int(request.form.get("writing_score")),
            reading_score=int(request.form.get("reading_score")),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])


if __name__ == "__main__":
    # by default port = 5000
    app.run(host="0.0.0.0")  # remove debug = true on deployment
