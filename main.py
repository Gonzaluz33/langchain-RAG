from flask import Flask
from flask_cors import CORS
from routes import routes_bp

app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "http://localhost:4200"}})
app.register_blueprint(routes_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


