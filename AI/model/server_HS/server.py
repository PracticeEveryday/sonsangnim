from flask_socketio import SocketIO, emit, disconnect
from flask import Flask
from flask_cors import CORS
from threading import Thread, Event

from model_load_HS import HandSignModel

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secret!'
socket = SocketIO(app, cors_allowed_origins='*', engineio_logger=True)


# Handle the webapp 

@socket.on('connect')
def connect_socket():
    print('user connected')
    

@socket.on('coordinate')
def handle_coordinate(data):
    model = HandSignModel()
    result = model.predict(data)
    emit("answer", result)


@socket.on("disconnect")
def disconnect_socket():
    # emit("disconnect", "ok")
    disconnect()


if __name__ == '__main__':
    socket.run(app, port=4000)
