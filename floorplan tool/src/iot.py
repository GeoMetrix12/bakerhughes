import os, json, time
from typing import Dict, Any, Optional

class IoTSink:
    def __init__(self, log_path: str = "outputs/iot/commands.log"):
        self.log_path = log_path
        self.mqtt = None
        self.topic_prefix = None
        broker = os.getenv("MQTT_BROKER")
        if broker:
            try:
                import paho.mqtt.client as mqtt
                self.mqtt = mqtt.Client()
                port = int(os.getenv("MQTT_PORT", "1883"))
                user = os.getenv("MQTT_USERNAME")
                pwd = os.getenv("MQTT_PASSWORD")
                if user and pwd:
                    self.mqtt.username_pw_set(user, pwd)
                self.mqtt.connect(broker, port, keepalive=30)
                self.topic_prefix = os.getenv("MQTT_TOPIC_PREFIX", "building/iot")
            except Exception:
                self.mqtt = None
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def send(self, floor_id: str, room_id: str, device: str, command: str, payload: Optional[Dict[str, Any]] = None) -> dict:
        msg = {
            "ts": time.time(),
            "floor_id": floor_id,
            "room_id": room_id,
            "device": device,
            "command": command,
            "payload": payload or {}
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        if self.mqtt:
            topic = f"{self.topic_prefix}/{floor_id}/{room_id}/{device}"
            try:
                self.mqtt.publish(topic, json.dumps(msg), qos=0, retain=False)
            except Exception:
                pass
        return msg
