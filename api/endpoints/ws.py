
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Websocket Real Time Push</title>
    </head>
    <body>
        <h1>Real Time Push</h1>
        <h2>Your Websocket ID: <span id="ws-id"></span></h2>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Get Real Time Data</button>
        </form>
        <h2>Please enter "start" and click the button</h2>
        <h2>The ID of websocket will change randomly, and each ID corresponds to different client data</h2>
        <ul id='messages'>
        </ul>
        <script>
            var client_id = Math.floor((Math.random()*1000)+1);
            document.querySelector("#ws-id").textContent = client_id;
            var ws = new WebSocket(`ws://localhost:8000/ws?coalYardId=10&clientId=${client_id}`);

            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""
