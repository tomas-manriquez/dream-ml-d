# Copyright (C) 2025 Leonardo Espinoza Ortiz <leonardo.espinoza.o@usach.cl>
#
# This file is part of DREAM ML.
#
# DREAM ML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DREAM ML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DREAM ML. If not, see <https://www.gnu.org/licenses/>.

import json
from channels.generic.websocket import AsyncWebsocketConsumer

class ProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Nos unimos al grupo "progreso_group"
        await self.channel_layer.group_add("progreso_group", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        # Al desconectar, removemos el canal del grupo
        await self.channel_layer.group_discard("progreso_group", self.channel_name)

    async def send_progress(self, event):
        # Este método se invoca cuando se envía un mensaje al grupo.
        # Se espera que el mensaje tenga los campos "step" y "status".
        step = event.get("step")
        status = event.get("status")
        # Enviamos al cliente WebSocket un mensaje JSON con la información
        await self.send(text_data=json.dumps({
            "step": step,
            "status": status
        }))
