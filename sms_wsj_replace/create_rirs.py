import numpy as np
from sms_wsj.reverb.reverb_utils import generate_rir
from sms_wsj.reverb.scenario import generate_random_source_positions
from sms_wsj.reverb.scenario import generate_sensor_positions
from sms_wsj.reverb.scenario import sample_from_random_box

def config():
    # Either set it to zero or above 0.15 s. Otherwise, RIR contains NaN.
    sound_decay_time_range = dict(low=0.15, high=0.6)

    geometry = dict(
        number_of_sources=3,
        number_of_sensors=7,
        sensor_shape="circular_center",
        center=[[3.5], [3.], [1.5]],    # m
        scale=0.0425,                   # m
        room=[[7.], [6.], [3.]],        # m
        random_box=[[4.], [2.], [0.4]],  # m
    )

    sample_rate = 16000
    filter_length = 2 ** 14  # 1.024 seconds when sample_rate == 16000

    return geometry, sound_decay_time_range, sample_rate, filter_length

def scenarios(geometry,sound_decay_time_range,src_position,):
    room_dimensions = sample_from_random_box(geometry["room"], geometry["random_box"])
    center = sample_from_random_box(geometry["center"], geometry["random_box"])
    source_positions = generate_random_source_positions(center=center,sources=geometry["number_of_sources"], dims=2)

    sensor_positions = generate_sensor_positions(
        shape=geometry["sensor_shape"],
        center=center,
        room_dimensions = room_dimensions,
        scale=geometry["scale"],
        number_of_sensors=geometry["number_of_sensors"],
        rotate_x=np.random.uniform(0, 0.01 * 2 * np.pi),
        rotate_y=np.random.uniform(0, 0.01 * 2 * np.pi),
        rotate_z=np.random.uniform(0, 2 * np.pi),
    )
    sound_decay_time = np.random.uniform(**sound_decay_time_range)

    return room_dimensions, source_positions, sensor_positions, sound_decay_time

def rirs(sample_rate, filter_length, room_dimensions, source_positions, sensor_positions, sound_decay_time):
    h = generate_rir(
        room_dimensions=room_dimensions,
        source_positions=source_positions,
        sensor_positions=sensor_positions,
        sound_decay_time=sound_decay_time,
        sample_rate=sample_rate,
        filter_length=filter_length,
        sensor_orientations=None,
        sensor_directivity=None,
        sound_velocity=343
    )

    return h
