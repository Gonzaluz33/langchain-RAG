from flask import Flask
from flask_cors import CORS
from routes import routes_bp
#from vectorstore import get_vectorstore, build_vectorstore_from_db

app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "http://localhost:4200"}})
app.register_blueprint(routes_bp)


if __name__ == "__main__":
    #build_vectorstore_from_db()
    app.run(host="0.0.0.0", port=8000)


