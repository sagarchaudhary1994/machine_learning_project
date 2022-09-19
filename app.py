from flask import Flask
from 

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    return "<p>CICD is done for this project</p>"


if __name__ == "__main__":
    app.run(debug=True)
