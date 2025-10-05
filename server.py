from flask import Flask, render_template, redirect, url_for
from flask_socketio import SocketIO, join_room, emit, request
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Track rematch requests per room
rematch_requests = {}

@app.route('/')
def index():
    # Redirect to a new game room
    room = str(uuid.uuid4())[:8]
    return redirect(url_for('game', room_id=room))

@app.route('/game/<room_id>')
def game(room_id):
    return render_template('connect4.html', room=room_id)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('message', {'msg': f"{data['player_name']} joined the room."}, room=room)

@socketio.on('move')
def on_move(data):
    room = data['room']
    move = data['move']
    emit('move', move, room=room, include_self=False)

@socketio.on('reset')
def on_reset(data):
    room = data['room']
    emit('reset', {}, room=room)

@socketio.on('rematch')
def on_rematch(data):
    room = data['room']
    sid = request.sid
    if room not in rematch_requests:
        rematch_requests[room] = set()
    rematch_requests[room].add(sid)
    # Find number of players in room
    players_in_room = socketio.server.manager.rooms['/'].get(room, set())
    if len(rematch_requests[room]) >= 2 or (len(players_in_room) <= 2 and len(rematch_requests[room]) == len(players_in_room)):
        emit('rematch', {}, room=room)
        rematch_requests[room] = set()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)