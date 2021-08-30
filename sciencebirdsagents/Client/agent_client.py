"""Agent to Server API"""

import json
import logging
import socket
import struct
import time
from enum import Enum

import cv2
import numpy as np
from PIL import Image


# logging.basicConfig(stream=sys.stdout, start_level=logging.DEBUG)
class GameState(Enum):
    """The state of the game at a particular instant"""
    UNKNOWN = 0
    MAIN_MENU = 1
    EPISODE_MENU = 2
    LEVEL_SELECTION = 3
    LOADING = 4
    PLAYING = 5
    WON = 6
    LOST = 7
    NEWTESTSET = 8
    NEWTRAININGSET = 9
    RESUMETRAINING = 10
    NEWTRIAL = 11
    REQUESTNOVELTYLIKELIHOOD = 12
    EVALUATION_TERMINATED = 13


class PlayingMode(Enum):
    """Mode of play"""
    COMPETITION = 0
    TRAINING = 1


class RequestCodes(Enum):
    """Codes for different requests"""
    DoScreenShot = 11
    Configure = 1
    SetGameSimulationSpeed = 2
    LoadLevel = 51
    RestartLevel = 52
    LoadNextAvailableLevel = 53
    Cshoot = 31
    Pshoot = 32
    GTshoot = 38
    GetState = 12
    FullyZoomOut = 34
    GetNoOfLevels = 15
    GetCurrentLevel = 14
    ShootSeq = 11
    CFastshoot = 41
    PFastshoot = 42
    ShootSeqFast = 43
    GetAllLevelScores = 23
    ClickInCentre = 36
    FullyZoomIn = 35
    GetGroundTruthWithScreenshot = 61
    GetGroundTruthWithoutScreenshot = 62
    GetNoisyGroundTruthWithScreenshot = 63
    GetNoisyGroundTruthWithoutScreenshot = 64
    GetCurrentLevelScore = 65
    ReportNoveltyLikelihood = 66
    ReportNoveltyDescription = 67
    ReadyForNewSet = 68
    NoveltyInfo = 69


class AgentClient:
    """Science Birds agent API"""

    def __init__(
            self,
            host,
            port,
            playing_mode=PlayingMode.TRAINING,
            **kwargs
    ):

        self.server_port = int(port)
        self.server_host = host
        self.playing_mode = playing_mode
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._buffer = bytearray()

        self._extra_args = kwargs
        if "logger" in kwargs:
            self._logger = kwargs['logger']
        else:
            self._logger = logging.getLogger('Agent Client')

        self._logger.setLevel(logging.CRITICAL)

    # @timeout_decorator.timeout(5, use_signals=False, timeout_exception=TimeoutError)
    def _read_raw_from_buff(self, size):
        """Read a specific number of bytes from server_socket"""
        self._logger.debug("Reading %s bytes from server", size)
        self.server_socket.settimeout(300)  # set timeout for 5 mins
        while len(self._buffer) < size:
            new_bytes = self.server_socket.recv(size - len(self._buffer))
            self._buffer.extend(new_bytes)
        encoded = bytearray(self._buffer[:size])
        self._logger.debug(
            "Read: |%s|",
            encoded.hex()[:75] + (encoded.hex()[75:] and "...")
        )
        self._buffer = self._buffer[size:]
        return encoded

    def _read_from_buff(self, fmt):
        """Read the struct fmt from server_socket"""
        fmt = "!" + fmt
        size = struct.calcsize(fmt)
        encoded = self._read_raw_from_buff(size)
        return struct.unpack(fmt, encoded)

    def _send_command(self, command, *args):
        """Send a command with formatted arguments to server"""
        fmt = args[0] if args else ""
        args = args[1:] if len(args) > 1 else []
        msg = bytearray(struct.pack("!B" + fmt, command.value, *args))
        self._logger.debug(
            "Sending Request %s with bytes: |%s|",
            command,
            msg.hex()[:75] + (msg.hex()[75:] and "...")
        )
        self.server_socket.sendall(msg)

    # INITIALIZATION
    def connect_to_server(self):
        try:
            self.server_socket.connect((self.server_host, self.server_port))
            self._logger.info(
                'Client connected to server on port: %d',
                self.server_port
            )
        except socket.error as e:
            self._logger.exception(
                'Client failed to connect to server.'
                + ' Requested HOST: %s'
                + ' Requested PORT: %d'
                + ' Error Message: %s',
                self.server_host, self.server_port, e)
            raise e

    def disconnect_from_server(self):
        try:
            self.server_socket.close()
            self._logger.info('Client disconnected from server.')
        except socket.error as e:
            self._logger.exception(
                'Client failed to disconnect from server.'
                + ' Requested HOST: %s'
                + ' Requested PORT: %d'
                + ' Error Message: %s',
                self.server_host, self.server_port, e)
            raise e

    # REQUESTS
    def configure(self, agent_id):
        """Send configure message to server"""
        self._logger.info("Sending configure request")
        self._send_command(
            RequestCodes.Configure,
            "IB",
            agent_id,
            self.playing_mode.value
        )

        (round_number, limit, levels) = self._read_from_buff("BBB")
        self._logger.info(
            'Received configuration: Round = %d, time_limit=%d, levels = %d',
            round_number, limit, levels
        )
        return (round_number, limit, levels)

    def ready_for_new_set(self):
        self._logger.info("Ready for new data set with appropriate agent.")
        self._send_command(RequestCodes.ReadyForNewSet)
        (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
         allowNoveltyInfo) = self._read_from_buff("IIIIBBB")
        return (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set, allowNoveltyInfo)

    def report_novelty_likelihood(self, report_novelty_likelihood, non_novelty_likelihood, id_array, novelty_level,
                                  novelty_description):
        self._logger.info("report novelty likelihood")

        id_array_length = len(id_array)
        encoded_description = novelty_description.encode('utf-8')
        des_bytearray = bytearray()
        des_bytearray.extend(encoded_description)
        msg_length = len(des_bytearray)

        self._send_command(RequestCodes.ReportNoveltyLikelihood,
                           "ffi" + str(id_array_length) + "iii" + str(msg_length) + "s", report_novelty_likelihood,
                           non_novelty_likelihood, id_array_length, *id_array, novelty_level, msg_length, des_bytearray)
        response = self._read_from_buff("B")[0]
        return response

    def report_novelty_description(self, novelty_description):
        self._logger.info("report novelty description")
        encoded_description = novelty_description.encode('utf-8')
        msg_length = len(encoded_description)
        self._send_command(RequestCodes.ReportNoveltyDescription, "I" + str(msg_length) + "s", msg_length,
                           encoded_description)
        response = self._read_from_buff("B")[0]
        return response

    def set_game_simulation_speed(self, simulation_speed):
        self._logger.info("Sending set simulation speed request")
        self._send_command(RequestCodes.SetGameSimulationSpeed, "I", simulation_speed)
        response = self._read_from_buff("B")[0]
        self._logger.info("Simulation speed is set to %d", simulation_speed)
        return response

    def read_image_from_stream(self, **kwargs):
        """Read image from server_socket"""
        (width, height) = self._read_from_buff("II")
        total_bytes = width * height * 3
        # Read the raw RGB data
        read_bytes = 0
        # read first bytes
        image_bytes = self.server_socket.recv(2048)
        read_bytes += image_bytes.__len__()

        # read the rest
        while (read_bytes < total_bytes):
            byte_buffer = self.server_socket.recv(2048)
            byte_buffer_length = byte_buffer.__len__()
            if (byte_buffer_length != -1):
                image_bytes += byte_buffer
            else:
                break
            read_bytes += byte_buffer_length

        rgb_image = Image.frombytes("RGB", (width, height), image_bytes)  # check if  PIL is needed
        # TODO: Remove after Debug
        # rgb_image.save(os.path.join('./', 'test'), format='png')

        self._logger.info('Received screenshot')

        img = np.array(rgb_image)
        # Convert RGB to BGR
        rgb_image = img[:, :, ::-1].copy()
        # cv2.imwrite('image.png',img)
        if 'resize' in kwargs:
            rgb_image = cv2.resize(rgb_image, (kwargs['resize'][1], kwargs['resize'][0]))
        img = np.transpose(rgb_image, (2, 0, 1))
        return img

    def read_ground_truth_from_stream(self):
        """Read Ground Truth fro sever_socket"""
        self._logger.debug("reading groundtruth from stream")
        msg_length = self._read_from_buff("I")[0]
        data = b''

        self._logger.debug("groundtruth length is %d bytes", msg_length)
        while len(data) < msg_length:
            packet = self.server_socket.recv(msg_length - len(data))
            if not packet:
                return None
            data += packet
        data_string = data.decode("UTF-8")
        data_string = data_string[:-5]
        return json.loads(data_string)

    def do_screenshot(self, **kwargs):
        """Request screenshot from server"""
        self._logger.info("Sending screenshot request")
        self._send_command(RequestCodes.DoScreenShot)
        return self.read_image_from_stream(**kwargs)

    def get_game_state(self):
        """Retrieve game state"""
        self._logger.info("Sending gamestate request")
        self._send_command(RequestCodes.GetState)
        state = GameState(self._read_from_buff("B")[0])
        self._logger.info("Got gamestate = %s", state)
        return state

    def load_level(self, level_number):
        """Load a specific start_level"""
        if level_number < 1:
            level_number = 1
        self._logger.info("Sending loadLevel request")
        self._send_command(RequestCodes.LoadLevel, "I", level_number)
        response = self._read_from_buff("B")[0]
        self._logger.info('Received loadLevel')
        return response

    def load_next_available_level(self):
        """Load the next available start_level"""
        self._logger.info("Sending load next available start_level request")
        self._send_command(RequestCodes.LoadNextAvailableLevel)
        level = self._read_from_buff("I")[0]
        self._logger.info('Received load next available start_level')
        return level

    def get_novelty_info(self):
        """query if novelty starts to appear"""
        self._send_command(RequestCodes.NoveltyInfo)
        novelty_info = self._read_from_buff("i")[0]
        self._logger.info("novelty existence is %d ", novelty_info)
        return novelty_info

    def shoot_and_record_ground_truth(self, fx, fy, t1, t2, gt_frequency):
        """ Request to execute a shot and record ground truth every gt_frequency frames
            Note: number of frames will be dependent on the set game simulation and gt_frequency
            the slower the game is -> more frequent ground truth snapshots are possible and vice verta.
        """
        start_time = time.time()

        code = RequestCodes.GTshoot
        should_read_images = False  # for now turned off completely on the server and SB, in case needed - ask
        self._send_command(code, "iiiii", fx, fy, t1, t2, gt_frequency)

        # read how many ground truths to expect
        ground_truths_count_bytes = self._read_from_buff("I")[0]
        ground_truths_count = int(ground_truths_count_bytes)

        # read n ground truths
        gt_images = []
        gt_jsons = []
        self._logger.info("receiving ground truth batch")
        for i in range(0, ground_truths_count):

            gt = self.read_ground_truth_from_stream()
            if (should_read_images):
                im = self.read_image_from_stream()
            if (i % 100 == 0):
                self._logger.info("received gt number %d", i)
            if (should_read_images):
                gt_images.append(im)
            gt_jsons.append(gt)
        self._logger.info("received %d ground truth frames ", ground_truths_count)
        self._logger.info("--- %s seconds ---", (time.time() - start_time))
        return gt_jsons

    def restart_level(self):
        """Request to restart start_level"""
        self._send_command(RequestCodes.RestartLevel)
        return self._read_from_buff("B")[0]

    def shoot(self, fx, fy, t1, t2, isPolar):
        code = RequestCodes.Pshoot if isPolar else RequestCodes.Cshoot
        self._send_command(code, "iiii", fx, fy, t1, t2)
        return self._read_from_buff("B")[0]

    def fast_shoot(self, fx, fy, t1, t2, isPolar):
        code = RequestCodes.PFastshoot if isPolar else RequestCodes.CFastshoot
        self._send_command(code, "iiii", fx, fy, t1, t2)
        return self._read_from_buff("B")[0]

    def get_all_level_scores(self):
        if self.playing_mode != PlayingMode.COMPETITION:
            self._logger.warning(
                "GetAllLevelScores is not recommended in %s",
                self.playing_mode
            )
        self._send_command(RequestCodes.GetAllLevelScores)
        n_levels = self._read_from_buff("I")[0]
        return self._read_from_buff("" + n_levels * "I")

    def get_current_score(self):
        self._send_command(RequestCodes.GetCurrentLevelScore)
        return self._read_from_buff("I")[0]

    def get_number_of_levels(self):
        self._logger.info("Requesting number of levels")
        self._send_command(RequestCodes.GetNoOfLevels)
        levels = self._read_from_buff("I")[0]
        self._logger.info("Number of Levels = %d", levels)
        return levels

    def get_current_level(self):
        self._send_command(RequestCodes.GetCurrentLevel)
        return self._read_from_buff("I")[0]

    def fully_zoom_in(self):
        self._send_command(RequestCodes.FullyZoomIn)
        return self._read_from_buff("B")[0]

    def fully_zoom_out(self):
        self._send_command(RequestCodes.FullyZoomOut)
        return self._read_from_buff("B")[0]

    def get_symbolic_state_with_screenshot(self):
        self._logger.info("sending get_symbolic_state_with_screenshot request")
        self._send_command(RequestCodes.GetGroundTruthWithScreenshot)
        gt = self.read_ground_truth_from_stream()
        im = self.read_image_from_stream()
        return (im, gt)

    def get_symbolic_state_without_screenshot(self, **kwargs):
        self._logger.info("sending get_symbolic_state_without_screenshot request")
        self._send_command(RequestCodes.GetGroundTruthWithoutScreenshot)
        return self.read_ground_truth_from_stream()

    def get_noisy_ground_truth_with_screenshot(self):
        self._logger.info("sending get_noisy_ground_truth_with_screenshot request")
        self._send_command(RequestCodes.GetNoisyGroundTruthWithScreenshot)
        gt = self.read_ground_truth_from_stream()
        im = self.read_image_from_stream()
        return (im, gt)

    def get_noisy_ground_truth_without_screenshot(self):
        self._logger.info("sending get_noisy_ground_truth_without_screenshot request")
        self._send_command(RequestCodes.GetNoisyGroundTruthWithoutScreenshot)
        gt = self.read_ground_truth_from_stream()
        return gt


if __name__ == "__main__":
    """ TEST AGENT """
    with open('./server_client_config.json', 'r') as config:
        sc_json_config = json.load(config)

    client = AgentClient(**sc_json_config[0])
    try:
        client.connect_to_server()
        client.configure(2888)
        img = client.do_screenshot()

        game_state = client.get_game_state()

        info = client.load_level(3)
        client.do_screenshot()
        level = client.get_current_level()

        client.fully_zoom_in()
        client.fully_zoom_out()
        info = client.shoot(172, 276, 943, 264, 0, 0, False)

        image, ground_truth = client.get_symbolic_state_with_screenshot()
        ground_truth = client.get_symbolic_state_without_screenshot()
        noisy_image, noisy_truth = client.get_noisy_ground_truth_with_screenshot()
        noisy_truth = client.get_noisy_ground_truth_without_screenshot()

        info = client.restart_level()
        client.disconnect_from_server()
    except socket.error as e:
        print("Error in client-server communication: " + str(e))
