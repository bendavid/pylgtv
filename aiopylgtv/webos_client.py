import asyncio
import base64
import copy
import json
import logging
import os
import time

import numpy as np
import websockets

from . import buttons as btn
from . import cal_commands as cal
from . import endpoints as ep
from .constants import CALIBRATION_TYPE_MAP, DEFAULT_CAL_DATA
from .handshake import REGISTRATION_MESSAGE
from .lut_tools import read_cal_file, read_cube_file, unity_lut_1d, unity_lut_3d

logger = logging.getLogger(__name__)


KEY_FILE_NAME = ".aiopylgtv"
USER_HOME = "HOME"


class PyLGTVPairException(Exception):
    def __init__(self, message):
        self.message = message


class PyLGTVCmdException(Exception):
    def __init__(self, message):
        self.message = message


class WebOsClient:
    def __init__(self, ip, key_file_path=None, timeout_connect=2, ping_interval=1):
        """Initialize the client."""
        self.ip = ip
        self.port = 3000
        self.key_file_path = key_file_path
        self.client_key = None
        self.web_socket = None
        self.command_count = 0
        self.timeout_connect = timeout_connect
        self.ping_interval = ping_interval
        self.connect_task = None
        self.connect_result = None
        self.connection = None
        self.input_connection = None
        self.callbacks = {}
        self.futures = {}
        self._power_state = {}
        self._current_appId = None
        self._muted = None
        self._volume = None
        self._current_channel = None
        self._channel_info = None
        self._channels = None
        self._apps = {}
        self._extinputs = {}
        self._system_info = None
        self._software_info = None
        self._sound_output = None
        self.state_update_callbacks = []
        self.doStateUpdate = False

        self.load_key_file()

    @staticmethod
    def _get_key_file_path():
        """Return the key file path."""
        if os.getenv(USER_HOME) is not None and os.access(
            os.getenv(USER_HOME), os.W_OK
        ):
            return os.path.join(os.getenv(USER_HOME), KEY_FILE_NAME)

        return os.path.join(os.getcwd(), KEY_FILE_NAME)

    def load_key_file(self):
        """Try to load the client key for the current ip."""
        self.client_key = None
        if self.key_file_path:
            key_file_path = self.key_file_path
        else:
            key_file_path = self._get_key_file_path()
        key_dict = {}

        logger.debug("load keyfile from %s", key_file_path)

        if os.path.isfile(key_file_path):
            with open(key_file_path, "r") as f:
                raw_data = f.read()
                if raw_data:
                    key_dict = json.loads(raw_data)

        logger.debug("getting client_key for %s from %s", self.ip, key_file_path)
        if self.ip in key_dict:
            self.client_key = key_dict[self.ip]

    def save_key_file(self):
        """Save the current client key."""
        if self.client_key is None:
            return

        if self.key_file_path:
            key_file_path = self.key_file_path
        else:
            key_file_path = self._get_key_file_path()

        logger.debug("save keyfile to %s", key_file_path)

        if os.path.isfile(key_file_path):
            with open(key_file_path, "r+") as f:
                raw_data = f.read()
                f.seek(0)
                f.truncate()
                key_dict = {}

                if raw_data:
                    key_dict = json.loads(raw_data)

                key_dict[self.ip] = self.client_key

                f.write(json.dumps(key_dict))
        else:
            with open(key_file_path, "w") as f:
                key_dict = {}
                key_dict[self.ip] = self.client_key
                f.write(json.dumps(key_dict))

    async def connect(self):
        if not self.is_connected():
            self.connect_result = asyncio.Future()
            self.connect_task = asyncio.create_task(
                self.connect_handler(self.connect_result)
            )
        return await self.connect_result

    async def disconnect(self):
        if self.is_connected():
            self.connect_task.cancel()
            try:
                await self.connect_task
            except asyncio.CancelledError:
                pass

    def is_registered(self):
        """Paired with the tv."""
        return self.client_key is not None

    def is_connected(self):
        return self.connect_task is not None and not self.connect_task.done()

    def registration_msg(self):
        handshake = copy.deepcopy(REGISTRATION_MESSAGE)
        handshake["payload"]["client-key"] = self.client_key
        return handshake

    async def connect_handler(self, res):

        handler_tasks = set()
        ws = None
        inputws = None
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    f"ws://{self.ip}:{self.port}",
                    ping_interval=None,
                    close_timeout=self.timeout_connect,
                    max_size=None,
                ),
                timeout=self.timeout_connect,
            )
            await ws.send(json.dumps(self.registration_msg()))
            raw_response = await ws.recv()
            response = json.loads(raw_response)

            if (
                response["type"] == "response"
                and response["payload"]["pairingType"] == "PROMPT"
            ):
                raw_response = await ws.recv()
                response = json.loads(raw_response)
                if response["type"] == "registered":
                    self.client_key = response["payload"]["client-key"]
                    self.save_key_file()

            if not self.client_key:
                raise PyLGTVPairException("Unable to pair")

            self.callbacks = {}
            self.futures = {}

            handler_tasks.add(
                asyncio.create_task(
                    self.consumer_handler(ws, self.callbacks, self.futures)
                )
            )
            if self.ping_interval is not None:
                handler_tasks.add(
                    asyncio.create_task(self.ping_handler(ws, self.ping_interval))
                )
            self.connection = ws

            # open additional connection needed to send button commands
            # the url is dynamically generated and returned from the ep.INPUT_SOCKET
            # endpoint on the main connection
            sockres = await self.request(ep.INPUT_SOCKET)
            inputsockpath = sockres.get("socketPath")
            inputws = await asyncio.wait_for(
                websockets.connect(
                    inputsockpath,
                    ping_interval=None,
                    close_timeout=self.timeout_connect,
                ),
                timeout=self.timeout_connect,
            )

            handler_tasks.add(asyncio.create_task(inputws.wait_closed()))
            if self.ping_interval is not None:
                handler_tasks.add(
                    asyncio.create_task(self.ping_handler(inputws, self.ping_interval))
                )
            self.input_connection = inputws

            # set static state and subscribe to state updates
            # avoid partial updates during initial subscription

            self.doStateUpdate = False
            self._system_info, self._software_info = await asyncio.gather(
                self.get_system_info(), self.get_software_info()
            )
            subscribe_coros = {
                self.subscribe_power_state(self.set_power_state),
                self.subscribe_current_app(self.set_current_app_state),
                self.subscribe_muted(self.set_muted_state),
                self.subscribe_volume(self.set_volume_state),
                self.subscribe_apps(self.set_apps_state),
                self.subscribe_inputs(self.set_inputs_state),
                self.subscribe_sound_output(self.set_sound_output_state),
            }
            subscribe_tasks = set()
            for coro in subscribe_coros:
                subscribe_tasks.add(asyncio.create_task(coro))
            subscribe_done, _ = await asyncio.wait(subscribe_tasks)
            for task in subscribe_done:
                try:
                    task.result()
                except PyLGTVCmdException:
                    pass
            # set placeholder power state if not available
            if not self._power_state:
                self._power_state = {"state": "Unknown"}
            self.doStateUpdate = True
            if self.state_update_callbacks:
                await self.do_state_update_callbacks()

            res.set_result(True)

            await asyncio.wait(handler_tasks, return_when=asyncio.FIRST_COMPLETED)

        except Exception as ex:
            if not res.done():
                res.set_exception(ex)
        finally:
            for task in handler_tasks:
                if not task.done():
                    task.cancel()

            for future in self.futures.values():
                future.cancel()

            closeout = set()
            closeout.update(handler_tasks)

            if ws is not None:
                closeout.add(asyncio.create_task(ws.close()))
            if inputws is not None:
                closeout.add(asyncio.create_task(inputws.close()))

            self.connection = None
            self.input_connection = None

            self.doStateUpdate = False

            self._power_state = {}
            self._current_appId = None
            self._muted = None
            self._volume = None
            self._current_channel = None
            self._channel_info = None
            self._channels = None
            self._apps = {}
            self._extinputs = {}
            self._system_info = None
            self._software_info = None
            self._sound_output = None

            for callback in self.state_update_callbacks:
                closeout.add(callback())

            if closeout:
                closeout_task = asyncio.create_task(asyncio.wait(closeout))

                while not closeout_task.done():
                    try:
                        await asyncio.shield(closeout_task)
                    except asyncio.CancelledError:
                        pass

    async def ping_handler(self, ws, interval):
        try:
            while True:
                await asyncio.sleep(interval)
                # In the "Suspend" state the tv can keep a connection alive, but will not respond to pings
                if self._power_state.get("state") != "Suspend":
                    ping_waiter = await ws.ping()
                    await asyncio.wait_for(ping_waiter, timeout=self.timeout_connect)
        except (
            asyncio.TimeoutError,
            asyncio.CancelledError,
            websockets.exceptions.ConnectionClosedError,
        ):
            pass

    async def callback_handler(self, queue, callback, future):
        try:
            while True:
                msg = await queue.get()
                payload = msg.get("payload")
                await callback(payload)
                if future is not None and not future.done():
                    future.set_result(msg)
        except asyncio.CancelledError:
            pass

    async def consumer_handler(self, ws, callbacks={}, futures={}):

        callback_queues = {}
        callback_tasks = {}

        try:
            async for raw_msg in ws:
                if callbacks or futures:
                    msg = json.loads(raw_msg)
                    uid = msg.get("id")
                    callback = self.callbacks.get(uid)
                    future = self.futures.get(uid)
                    if callback is not None:
                        if uid not in callback_tasks:
                            queue = asyncio.Queue()
                            callback_queues[uid] = queue
                            callback_tasks[uid] = asyncio.create_task(
                                self.callback_handler(queue, callback, future)
                            )
                        callback_queues[uid].put_nowait(msg)
                    elif future is not None and not future.done():
                        self.futures[uid].set_result(msg)

        except (websockets.exceptions.ConnectionClosedError, asyncio.CancelledError):
            pass
        finally:
            for task in callback_tasks.values():
                if not task.done():
                    task.cancel()

            tasks = set()
            tasks.update(callback_tasks.values())

            if tasks:
                closeout_task = asyncio.create_task(asyncio.wait(tasks))

            while not closeout_task.done():
                try:
                    await asyncio.shield(closeout_task)
                except asyncio.CancelledError:
                    pass

    # manage state
    @property
    def power_state(self):
        return self._power_state

    @property
    def current_appId(self):
        return self._current_appId

    @property
    def muted(self):
        return self._muted

    @property
    def volume(self):
        return self._volume

    @property
    def current_channel(self):
        return self._current_channel

    @property
    def channel_info(self):
        return self._channel_info

    @property
    def channels(self):
        return self._channels

    @property
    def apps(self):
        return self._apps

    @property
    def inputs(self):
        return self._extinputs

    @property
    def system_info(self):
        return self._system_info

    @property
    def software_info(self):
        return self._software_info

    @property
    def sound_output(self):
        return self._sound_output

    @property
    def is_on(self):
        state = self._power_state.get("state")
        if state == "Unknown":
            # fallback to current app id for some older webos versions which don't support explicit power state
            if self._current_appId in [None, ""]:
                return False
            else:
                return True
        elif state in [None, "Power Off", "Suspend", "Active Standby"]:
            return False
        else:
            return True

    def calibration_support_info(self):
        info = {
            "lut1d": False,
            "lut3d_size": None,
            "custom_tone_mapping": False,
            "dv_config_type": None,
        }
        model_name = self._system_info["modelName"]
        if model_name.startswith("OLED") and len(model_name) > 7:
            model = model_name[6]
            year = int(model_name[7])
            if year >= 8:
                info["lut1d"] = True
                if model == "B":
                    info["lut3d_size"] = 17
                else:
                    info["lut3d_size"] = 33
            if year == 8:
                info["dv_config_type"] = 2018
            elif year == 9:
                info["custom_tone_mapping"] = True
                info["dv_config_type"] = 2019
        elif len(model_name) > 5:
            size = None
            try:
                size = int(model_name[0:2])
            except ValueError:
                pass
            if size:
                modeltype = model_name[2]
                modelyear = model_name[3]
                modelseries = model_name[4]
                modelnumber = model_name[5]

                if modeltype == "S" and modelyear in ["K", "M"] and modelseries >= 8:
                    info["lut1d"] = True
                    if modelseries == 9 and modelnumber == 9:
                        info["lut3d_size"] = 33
                    else:
                        info["lut3d_size"] = 17
                    if modelyear == "K":
                        info["dv_config_type"] = 2018
                    elif modelyear == "M":
                        info["custom_tone_mapping"] = True
                        info["dv_config_type"] = 2019

        return info

    async def register_state_update_callback(self, callback):
        self.state_update_callbacks.append(callback)
        if self.doStateUpdate:
            await callback()

    def unregister_state_update_callback(self, callback):
        if callback in self.state_update_callbacks:
            self.state_update_callbacks.remove(callback)

    def clear_state_update_callbacks(self):
        self.state_update_callbacks = []

    async def do_state_update_callbacks(self):
        callbacks = set()
        for callback in self.state_update_callbacks:
            callbacks.add(callback())

        if callbacks:
            await asyncio.gather(*callbacks)

    async def set_power_state(self, payload):
        self._power_state = payload

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_current_app_state(self, appId):
        """Set current app state variable.  This function also handles subscriptions to current channel and channel list, since the current channel subscription can only succeed when Live TV is running, and the channel list subscription can only succeed after channels have been configured."""
        self._current_appId = appId

        if self._channels is None:
            try:
                await self.subscribe_channels(self.set_channels_state)
            except PyLGTVCmdException:
                pass

        if appId == "com.webos.app.livetv":
            try:
                time.sleep(1)
                await self.subscribe_current_channel(self.set_current_channel_state)
            except PyLGTVCmdException:
                pass

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_muted_state(self, muted):
        self._muted = muted

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_volume_state(self, volume):
        self._volume = volume

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_channels_state(self, channels):
        self._channels = channels

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_current_channel_state(self, channel):
        """Set current channel state variable.  This function also handles the channel info subscription, since that call may fail if channel information is not available when it's called."""

        self._current_channel = channel

        if self._channel_info is None:
            try:
                await self.subscribe_channel_info(self.set_channel_info_state)
            except PyLGTVCmdException:
                pass

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_channel_info_state(self, channel_info):
        self._channel_info = channel_info

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_apps_state(self, apps):
        self._apps = {}
        for app in apps:
            self._apps[app["id"]] = app

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_inputs_state(self, extinputs):
        self._extinputs = {}
        for extinput in extinputs:
            self._extinputs[extinput["appId"]] = extinput

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    async def set_sound_output_state(self, sound_output):
        self._sound_output = sound_output

        if self.state_update_callbacks and self.doStateUpdate:
            await self.do_state_update_callbacks()

    # low level request handling

    async def command(self, request_type, uri, payload=None, uid=None):
        """Build and send a command."""
        if uid is None:
            uid = self.command_count
            self.command_count += 1

        if payload is None:
            payload = {}

        message = {
            "id": uid,
            "type": request_type,
            "uri": f"ssap://{uri}",
            "payload": payload,
        }

        if self.connection is None:
            raise PyLGTVCmdException("Not connected, can't execute command.")

        await self.connection.send(json.dumps(message))

    async def request(self, uri, payload=None, cmd_type="request", uid=None):
        """Send a request and wait for response."""
        if uid is None:
            uid = self.command_count
            self.command_count += 1
        res = asyncio.Future()
        self.futures[uid] = res
        try:
            await self.command(cmd_type, uri, payload, uid)
        except (asyncio.CancelledError, PyLGTVCmdException):
            del self.futures[uid]
            raise
        try:
            response = await res
        except asyncio.CancelledError:
            if uid in self.futures:
                del self.futures[uid]
            raise
        del self.futures[uid]

        payload = response.get("payload")
        if payload is None:
            raise PyLGTVCmdException(f"Invalid request response {response}")

        returnValue = payload.get("returnValue") or payload.get("subscribed")

        if returnValue is None:
            raise PyLGTVCmdException(f"Invalid request response {response}")
        elif not returnValue:
            raise PyLGTVCmdException(f"Request failed with response {response}")

        return payload

    async def subscribe(self, callback, uri, payload=None):
        """Subscribe to updates."""
        uid = self.command_count
        self.command_count += 1
        self.callbacks[uid] = callback
        try:
            return await self.request(
                uri, payload=payload, cmd_type="subscribe", uid=uid
            )
        except Exception:
            del self.callbacks[uid]
            raise

    async def input_command(self, message):
        if self.input_connection is None:
            raise PyLGTVCmdException("Couldn't execute input command.")

        await self.input_connection.send(message)

    # high level request handling

    async def button(self, name):
        """Send button press command."""

        message = f"type:button\nname:{name}\n\n"
        await self.input_command(message)

    async def move(self, dx, dy, down=0):
        """Send cursor move command."""

        message = f"type:move\ndx:{dx}\ndy:{dy}\ndown:{down}\n\n"
        await self.input_command(message)

    async def click(self):
        """Send cursor click command."""

        message = f"type:click\n\n"
        await self.input_command(message)

    async def scroll(self, dx, dy):
        """Send scroll command."""

        message = f"type:scroll\ndx:{dx}\ndy:{dy}\n\n"
        await self.input_command(message)

    async def send_message(self, message, icon_path=None):
        """Show a floating message."""
        icon_encoded_string = ""
        icon_extension = ""

        if icon_path is not None:
            icon_extension = os.path.splitext(icon_path)[1][1:]
            with open(icon_path, "rb") as icon_file:
                icon_encoded_string = base64.b64encode(icon_file.read()).decode("ascii")

        return await self.request(
            ep.SHOW_MESSAGE,
            {
                "message": message,
                "iconData": icon_encoded_string,
                "iconExtension": icon_extension,
            },
        )

    async def get_power_state(self):
        """Get current power state."""
        return await self.request(ep.GET_POWER_STATE)

    async def subscribe_power_state(self, callback):
        """Subscribe to current power state."""
        return await self.subscribe(callback, ep.GET_POWER_STATE)

    # Apps
    async def get_apps(self):
        """Return all apps."""
        res = await self.request(ep.GET_APPS)
        return res.get("launchPoints")

    async def subscribe_apps(self, callback):
        """Subscribe to changes in available apps."""

        async def apps(payload):
            await callback(payload.get("launchPoints"))

        return await self.subscribe(apps, ep.GET_APPS)

    async def get_current_app(self):
        """Get the current app id."""
        res = await self.request(ep.GET_CURRENT_APP_INFO)
        return res.get("appId")

    async def subscribe_current_app(self, callback):
        """Subscribe to changes in the current app id."""

        async def current_app(payload):
            await callback(payload.get("appId"))

        return await self.subscribe(current_app, ep.GET_CURRENT_APP_INFO)

    async def launch_app(self, app):
        """Launch an app."""
        return await self.request(ep.LAUNCH, {"id": app})

    async def launch_app_with_params(self, app, params):
        """Launch an app with parameters."""
        return await self.request(ep.LAUNCH, {"id": app, "params": params})

    async def launch_app_with_content_id(self, app, contentId):
        """Launch an app with contentId."""
        return await self.request(ep.LAUNCH, {"id": app, "contentId": contentId})

    async def close_app(self, app):
        """Close the current app."""
        return await self.request(ep.LAUNCHER_CLOSE, {"id": app})

    # Services
    async def get_services(self):
        """Get all services."""
        res = await self.request(ep.GET_SERVICES)
        return res.get("services")

    async def get_software_info(self):
        """Return the current software status."""
        return await self.request(ep.GET_SOFTWARE_INFO)

    async def get_system_info(self):
        """Return the system information."""
        return await self.request(ep.GET_SYSTEM_INFO)

    async def power_off(self):
        """Power off TV."""

        # protect against turning tv back on if it is off
        if not self.is_on:
            return

        # if tv is shutting down and standby+ option is not enabled,
        # response is unreliable, so don't wait for one,
        await self.command("request", ep.POWER_OFF)

    async def power_on(self):
        """Play media."""
        return await self.request(ep.POWER_ON)

    # 3D Mode
    async def turn_3d_on(self):
        """Turn 3D on."""
        return await self.request(ep.SET_3D_ON)

    async def turn_3d_off(self):
        """Turn 3D off."""
        return await self.request(ep.SET_3D_OFF)

    # Inputs
    async def get_inputs(self):
        """Get all inputs."""
        res = await self.request(ep.GET_INPUTS)
        return res.get("devices")

    async def subscribe_inputs(self, callback):
        """Subscribe to changes in available inputs."""

        async def inputs(payload):
            await callback(payload.get("devices"))

        return await self.subscribe(inputs, ep.GET_INPUTS)

    async def get_input(self):
        """Get current input."""
        return await self.get_current_app()

    async def set_input(self, input):
        """Set the current input."""
        return await self.request(ep.SET_INPUT, {"inputId": input})

    # Audio
    async def get_audio_status(self):
        """Get the current audio status"""
        return await self.request(ep.GET_AUDIO_STATUS)

    async def get_muted(self):
        """Get mute status."""
        status = await self.get_audio_status()
        return status.get("mute")

    async def subscribe_muted(self, callback):
        """Subscribe to changes in the current mute status."""

        async def muted(payload):
            await callback(payload.get("mute"))

        return await self.subscribe(muted, ep.GET_AUDIO_STATUS)

    async def set_mute(self, mute):
        """Set mute."""
        return await self.request(ep.SET_MUTE, {"mute": mute})

    async def get_volume(self):
        """Get the current volume."""
        res = await self.request(ep.GET_VOLUME)
        return res.get("volume")

    async def subscribe_volume(self, callback):
        """Subscribe to changes in the current volume."""

        async def volume(payload):
            await callback(payload.get("volume"))

        return await self.subscribe(volume, ep.GET_VOLUME)

    async def set_volume(self, volume):
        """Set volume."""
        volume = max(0, volume)
        return await self.request(ep.SET_VOLUME, {"volume": volume})

    async def volume_up(self):
        """Volume up."""
        return await self.request(ep.VOLUME_UP)

    async def volume_down(self):
        """Volume down."""
        return await self.request(ep.VOLUME_DOWN)

    # TV Channel
    async def channel_up(self):
        """Channel up."""
        return await self.request(ep.TV_CHANNEL_UP)

    async def channel_down(self):
        """Channel down."""
        return await self.request(ep.TV_CHANNEL_DOWN)

    async def get_channels(self):
        """Get list of tv channels."""
        res = await self.request(ep.GET_TV_CHANNELS)
        return res.get("channelList")

    async def subscribe_channels(self, callback):
        """Subscribe to list of tv channels."""

        async def channels(payload):
            await callback(payload.get("channelList"))

        return await self.subscribe(channels, ep.GET_TV_CHANNELS)

    async def get_current_channel(self):
        """Get the current tv channel."""
        return await self.request(ep.GET_CURRENT_CHANNEL)

    async def subscribe_current_channel(self, callback):
        """Subscribe to changes in the current tv channel."""
        return await self.subscribe(callback, ep.GET_CURRENT_CHANNEL)

    async def get_channel_info(self):
        """Get the current channel info."""
        return await self.request(ep.GET_CHANNEL_INFO)

    async def subscribe_channel_info(self, callback):
        """Subscribe to current channel info."""
        return await self.subscribe(callback, ep.GET_CHANNEL_INFO)

    async def set_channel(self, channel):
        """Set the current channel."""
        return await self.request(ep.SET_CHANNEL, {"channelId": channel})

    async def get_sound_output(self):
        """Get the current audio output."""
        res = await self.request(ep.GET_SOUND_OUTPUT)
        return res.get("soundOutput")

    async def subscribe_sound_output(self, callback):
        """Subscribe to changes in current audio output."""

        async def sound_output(payload):
            await callback(payload.get("soundOutput"))

        return await self.subscribe(sound_output, ep.GET_SOUND_OUTPUT)

    async def change_sound_output(self, output):
        """Change current audio output."""
        return await self.request(ep.CHANGE_SOUND_OUTPUT, {"output": output})

    # Media control
    async def play(self):
        """Play media."""
        return await self.request(ep.MEDIA_PLAY)

    async def pause(self):
        """Pause media."""
        return await self.request(ep.MEDIA_PAUSE)

    async def stop(self):
        """Stop media."""
        return await self.request(ep.MEDIA_STOP)

    async def close(self):
        """Close media."""
        return await self.request(ep.MEDIA_CLOSE)

    async def rewind(self):
        """Rewind media."""
        return await self.request(ep.MEDIA_REWIND)

    async def fast_forward(self):
        """Fast Forward media."""
        return await self.request(ep.MEDIA_FAST_FORWARD)

    # Keys
    async def send_enter_key(self):
        """Send enter key."""
        return await self.request(ep.SEND_ENTER)

    async def send_delete_key(self):
        """Send delete key."""
        return await self.request(ep.SEND_DELETE)

    # Web
    async def open_url(self, url):
        """Open URL."""
        return await self.request(ep.OPEN, {"target": url})

    async def close_web(self):
        """Close web app."""
        return await self.request(ep.CLOSE_WEB_APP)

    # Emulated button presses
    async def left_button(self):
        """left button press."""
        await self.button(btn.LEFT)

    async def right_button(self):
        """right button press."""
        await self.button(btn.RIGHT)

    async def down_button(self):
        """down button press."""
        await self.button(btn.DOWN)

    async def up_button(self):
        """up button press."""
        await self.button(btn.UP)

    async def home_button(self):
        """home button press."""
        await self.button(btn.HOME)

    async def back_button(self):
        """back button press."""
        await self.button(btn.BACK)

    async def ok_button(self):
        """ok button press."""
        await self.button(btn.ENTER)

    async def dash_button(self):
        """dash button press."""
        await self.button(btn.DASH)

    async def info_button(self):
        """info button press."""
        await self.button(btn.INFO)

    async def asterisk_button(self):
        """asterisk button press."""
        await self.button(btn.ASTERISK)

    async def cc_button(self):
        """cc button press."""
        await self.button(btn.CC)

    async def exit_button(self):
        """exit button press."""
        await self.button(btn.EXIT)

    async def mute_button(self):
        """mute button press."""
        await self.button(btn.MUTE)

    async def red_button(self):
        """red button press."""
        await self.button(btn.RED)

    async def green_button(self):
        """green button press."""
        await self.button(btn.GREEN)

    async def blue_button(self):
        """blue button press."""
        await self.button(btn.BLUE)

    async def volume_up_button(self):
        """volume up button press."""
        await self.button(btn.VOLUMEUP)

    async def volume_down_button(self):
        """volume down button press."""
        await self.button(btn.VOLUMEDOWN)

    async def channel_up_button(self):
        """channel up button press."""
        await self.button(btn.CHANNELUP)

    async def channel_down_button(self):
        """channel down button press."""
        await self.button(btn.CHANNELDOWN)

    async def play_button(self):
        """play button press."""
        await self.button(btn.PLAY)

    async def pause_button(self):
        """pause button press."""
        await self.button(btn.PAUSE)

    async def number_button(self, num):
        """numeric button press."""
        if not (num >= 0 and num <= 9):
            raise ValueError

        await self.button(f"""{num}""")

    def validateCalibrationData(self, data, shape, dtype):
        if not isinstance(data, np.ndarray):
            raise TypeError
        if data.shape != shape:
            raise ValueError
        if data.dtype != dtype:
            raise TypeError

    async def calibration_request(self, command, picMode, data):
        dataenc = base64.b64encode(data.tobytes()).decode()

        payload = {
            "command": command,
            "data": dataenc,
            "dataCount": data.size,
            "dataOpt": 1,
            "dataType": CALIBRATION_TYPE_MAP[data.dtype.name],
            "profileNo": 0,
            "programID": 1,
            "picMode": picMode,
        }

        return await self.request(ep.CALIBRATION, payload)

    async def start_calibration(self, picMode, data=DEFAULT_CAL_DATA):
        self.validateCalibrationData(data, (9,), np.float32)
        return await self.calibration_request(cal.CAL_START, picMode, data)

    async def end_calibration(self, picMode, data=DEFAULT_CAL_DATA):
        self.validateCalibrationData(data, (9,), np.float32)
        return await self.calibration_request(cal.CAL_END, picMode, data)

    async def upload_1d_lut(self, picMode, data=None):
        info = self.calibration_support_info()
        if not info["lut1d"]:
            model = self._system_info["modelName"]
            raise PyLGTVCmdException(
                f"1D LUT Upload not supported by tv model {model}."
            )
        if data is None:
            data = unity_lut_1d()
        self.validateCalibrationData(data, (3, 1024), np.uint16)
        return await self.calibration_request(cal.UPLOAD_1D_LUT, picMode, data)

    async def upload_3d_lut(self, command, picMode, data):
        if command not in [cal.UPLOAD_3D_LUT_BT709, cal.UPLOAD_3D_LUT_BT2020]:
            raise PyLGTVCmdException(f"Invalid 3D LUT Upload command {command}.")
        info = self.calibration_support_info()
        lut3d_size = info["lut3d_size"]
        if not lut3d_size:
            model = self._system_info["modelName"]
            raise PyLGTVCmdException(
                f"3D LUT Upload not supported by tv model {model}."
            )
        if data is None:
            data = unity_lut_3d(lut3d_size)
        lut3d_shape = (lut3d_size, lut3d_size, lut3d_size, 3)
        self.validateCalibrationData(data, lut3d_shape, np.uint16)
        return await self.calibration_request(command, picMode, data)

    async def upload_3d_lut_bt709(self, picMode, data=None):
        return await self.upload_3d_lut(cal.UPLOAD_3D_LUT_BT709, picMode, data)

    async def upload_3d_lut_bt2020(self, picMode, data=None):
        return await self.upload_3d_lut(cal.UPLOAD_3D_LUT_BT2020, picMode, data)

    async def set_ui_data(self, command, picMode, value):
        if isinstance(value, str):
            value = int(value)

        if not (value >= 0 and value <= 100):
            raise ValueError

        data = np.array(value, dtype=np.uint16)
        return await self.calibration_request(command, picMode, data)

    async def set_brightness(self, picMode, value):
        return await self.set_ui_data(cal.BRIGHTNESS_UI_DATA, picMode, value)

    async def set_contrast(self, picMode, value):
        return await self.set_ui_data(cal.CONTRAST_UI_DATA, picMode, value)

    async def set_oled_light(self, picMode, value):
        return await self.set_ui_data(cal.BACKLIGHT_UI_DATA, picMode, value)

    async def set_color(self, picMode, value):
        return await self.set_ui_data(cal.COLOR_UI_DATA, picMode, value)

    async def set_1d_2_2_en(self, picMode, value=0):
        data = np.array(value, dtype=np.uint16)
        return await self.calibration_request(
            cal.ENABLE_GAMMA_2_2_TRANSFORM, picMode, data
        )

    async def set_1d_0_45_en(self, picMode, value=0):
        data = np.array(value, dtype=np.uint16)
        return await self.calibration_request(
            cal.ENABLE_GAMMA_0_45_TRANSFORM, picMode, data
        )

    async def set_bt709_3by3_gamut_data(
        self, picMode, data=np.identity(3, dtype=np.float32)
    ):
        self.validateCalibrationData(data, (3, 3), np.float32)
        return await self.calibration_request(cal.BT709_3BY3_GAMUT_DATA, picMode, data)

    async def set_bt2020_3by3_gamut_data(
        self, picMode, data=np.identity(3, dtype=np.float32)
    ):
        self.validateCalibrationData(data, (3, 3), np.float32)
        return await self.calibration_request(cal.BT2020_3BY3_GAMUT_DATA, picMode, data)

    async def set_tonemap_params(
        self,
        picMode,
        luminance=700,
        mastering_peak_1=1000,
        rolloff_point_1=70,
        mastering_peak_2=4000,
        rolloff_point_2=60,
        mastering_peak_3=10000,
        rolloff_point_3=50,
    ):

        data = np.array(
            [
                luminance,
                mastering_peak_1,
                rolloff_point_1,
                mastering_peak_2,
                rolloff_point_2,
                mastering_peak_3,
                rolloff_point_3,
            ],
            dtype=np.uint16,
        )

        return await self.calibration_request(cal.SET_TONEMAP_PARAM, picMode, data)

    async def ddc_reset(self, picMode, reset_1d_lut=True):
        if isinstance(reset_1d_lut, str):
            if reset_1d_lut.lower() == "true":
                reset_1d_lut = True
            elif reset_1d_lut.lower() == "false":
                reset_1d_lut = False
            else:
                try:
                    reset_1d_lut = bool(int(reset_1d_lut))
                except ValueError:
                    raise ValueError(
                        f"Invalid parameter {reset_1d_lut} for ddc_reset, should be a boolean."
                    )

        await self.set_1d_2_2_en(picMode)
        await self.set_1d_0_45_en(picMode)
        await self.set_bt709_3by3_gamut_data(picMode)
        await self.set_bt2020_3by3_gamut_data(picMode)
        await self.upload_3d_lut_bt709(picMode)
        await self.upload_3d_lut_bt2020(picMode)
        if reset_1d_lut:
            await self.upload_1d_lut(picMode)

        return True

    async def get_picture_settings(
        self, keys=["contrast", "backlight", "brightness", "color"]
    ):
        payload = {"category": "picture", "keys": keys}
        ret = await self.request(ep.GET_SYSTEM_SETTINGS, payload=payload)
        return ret["settings"]

    async def upload_1d_lut_from_file(self, picMode, filename):
        ext = filename.split(".")[-1].lower()
        if ext == "cal":
            lut = read_cal_file(filename)
        elif ext == "cube":
            lut = read_cube_file(filename)
        else:
            raise ValueError(
                f"Unsupported file format {ext} for 1D LUT.  Supported file formats are cal and cube."
            )

        return await self.upload_1d_lut(picMode, lut)

    async def upload_3d_lut_from_file(self, command, picMode, filename):
        ext = filename.split(".")[-1].lower()
        if ext == "cube":
            lut = read_cube_file(filename)
        else:
            raise ValueError(
                f"Unsupported file format {ext} for 3D LUT.  Supported file formats are cube."
            )

        return await self.upload_3d_lut(command, picMode, lut)

    async def upload_3d_lut_bt709_from_file(self, picMode, filename):
        return await self.upload_3d_lut_from_file(
            cal.UPLOAD_3D_LUT_BT709, picMode, filename
        )

    async def upload_3d_lut_bt2020_from_file(self, picMode, filename):
        return await self.upload_3d_lut_from_file(
            cal.UPLOAD_3D_LUT_BT2020, picMode, filename
        )
