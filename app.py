from flask import Flask

app=Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    return "<p>Starting Machine Learning Project</p>"


if __name__=="__main__":
    app.run(debug=True)