import functools
import logging
from logstash_async.handler import AsynchronousLogstashHandler
import io
import streamlit as st
import yaml
from minio import Minio
from process import *
from get_logger import *
import consul
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
import os
import matplotlib.pyplot as plt


def get_config():
    with open('config.yaml', 'rb') as f:
        cfg = yaml.safe_load(f)
    return cfg


account = get_config()["account"]

cfg_minio = get_config()["minio"]

client = Minio(endpoint=cfg_minio["endpoint"], access_key=cfg_minio["access_key"], secret_key=cfg_minio["secret_key"],
               secure=False)


def get_consult():
    return consul.Consul(host='localhost', port=80, scheme='http', verify=False)


class UI:
    def run(self): pass


class ContributeData(UI):
    name = 'Contribute Data Service'

    def __init__(self):
        c = get_consult()
        self.file = st.file_uploader("Upload file", accept_multiple_files=True)
        self.button = st.button("Send")

    def run(self):
        if self.button:
            if self.file is None:
                st.error('File is empty')
            else:
                for f in self.file:
                    client.put_object(bucket_name="breath-data", object_name=f.name, data=io.BytesIO(f.read()),
                                      length=f.size)
            st.success("Send file successful")


class BreathDetection(UI):
    name = "Breath Detection Service"

    def __init__(self):
        c = get_consult()
        self.user = st.text_input("ID person")
        self.file = None
        self.dir_file = 'overlapped_files/deep/'
        self.record_file = 'output/'
        self.model_file = 'pretrained_model/model.hdf5'
        self.choose_box = st.selectbox('Choose type', ['Upload file', 'Record file'])

    def run(self):
        if self.choose_box == 'Record file':
            with st.spinner(f'Recording for {DURATION} seconds ....'):
                sound.record()
            st.success("Recording completed")
            self.file = 'output/recording/recorded.wav'
            split_file(self.file, self.dir_file)


        else:
            self.file = st.file_uploader('File audio')
            button = st.button("Detect")
            if button:
                if len(self.user) == 0:
                    st.error("Id must not empty")
                elif not self.user.lower().startswith("per"):
                    st.error("Id must start with PER/per")
                else:
                    if self.file is None:
                        st.error('File is empty')
                    elif not str(self.file.name).endswith('.wav'):
                        st.error('File must be wav file')
                    else:
                        st.write('')
                        st.write('')
                        st.write('')
                        st.write('**Detected Result**')
                        cols = st.beta_columns(2)
                        split_file(self.file, self.dir_file)
                        self.records = classify('overlapped_files',self.model_file)
                        plot_point(plt, self.records, len(self.records), cols[0])
                        get_report(self.records, len(self.records), cols[1], self.user)
                        test_logger = get_log()
                        total_duration, duration = get_extra(self.records)
                        extra = {
                            "per_id": self.user,
                            "duration": total_duration,
                            "normal": duration[0],
                            "deep": duration[1],
                            "strong": duration[2],
                            "other": duration[3]
                        }
                        test_logger.info('Report Result', extra=extra)


class LogManagement(UI):
    name = "Breath Tracking Service"

    def __init__(self):
        c = get_consult()
        self.link = 'http://localhost:5601/app/discover#'

    def run(self):
        st.write('**Follow this link**')
        st.markdown(self.link, unsafe_allow_html=True)


def get_subclasses(cls):
    """returns all subclasses of argument, cls"""
    if issubclass(cls, type):
        subclasses = cls.__subclasses__(cls)
    else:
        subclasses = cls.__subclasses__()
    for subclass in subclasses:
        subclasses.extend(get_subclasses(subclass))
    return subclasses


def cache_on_button_press(label, **cache_kwargs):
    """Function decorator to memoize function executions.

    Parameters
    ----------
    label : str
        The label for the button to display prior to running the cached funnction.
    cache_kwargs : Dict[Any, Any]
        Additional parameters (such as show_spinner) to pass into the underlying @st.cache decorator.
    """
    internal_cache_kwargs = dict(cache_kwargs)
    internal_cache_kwargs['allow_output_mutation'] = True
    internal_cache_kwargs['show_spinner'] = False

    def function_decorator(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            @st.cache(**internal_cache_kwargs)
            def get_cache_entry(func, args, kwargs):
                class ButtonCacheEntry:
                    def __init__(self):
                        self.evaluated = False
                        self.return_value = None

                    def evaluate(self):
                        self.evaluated = True
                        self.return_value = func(*args, **kwargs)

                return ButtonCacheEntry()

            cache_entry = get_cache_entry(func, args, kwargs)
            if not cache_entry.evaluated:
                if st.sidebar.button(label):
                    cache_entry.evaluate()
                else:
                    raise st.StopException
            return cache_entry.return_value

        return wrapped_func

    return function_decorator


st.sidebar.subheader('Login Page')
username = st.sidebar.text_input('Username')
password = st.sidebar.text_input('Password', '******', type="password")


@cache_on_button_press('Authenticate')
def authenticate(username, password):
    return username in account and account[username] == password


if authenticate(username, password):
    all_class = {x.name: x for x in get_subclasses(UI)}
    arr = list(all_class.keys())
    task = st.sidebar.selectbox('Choose service', arr)
    t = all_class[task]
    e = t()
    e.run()
else:
    st.sidebar.error('Authentication failed')
