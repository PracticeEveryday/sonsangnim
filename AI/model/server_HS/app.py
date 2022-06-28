from flask_socketio import SocketIO, emit, disconnect
from flask import Flask
from flask_cors import CORS

from model_load_HS import HandSignModel

socket = SocketIO()


def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['SECRET_KEY'] = 'secret!'

    socket.init_app(app, async_mode='eventlet', cors_allowed_origins='*', engineio_logger=True)

    # Handle the webapp
    @app.route("/")
    def index():
        print('hi flask')
        return "hello world"

    @socket.on('connect')
    def connect_socket():
        print('user connected')

    @socket.on('coordinate')
    def handle_coordinate(data):
        model = HandSignModel('A')
        result = model.predict(data)
        emit("answer", result)

    @socket.on("disconnect")
    def disconnect_socket():
        # emit("disconnect", "ok")
        disconnect()

    return app
