import pandas
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import streamlit as st
import torch
from bokeh.plotting import figure
from network import EEG_Denoise

file = None
file_data = None

model = EEG_Denoise()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('EEG_Denoise' + '.pth'))

def split_arr(arr):
    return [arr[num:num+512] / 10 for num in range(0, len(arr), 512)]


def predict(data, num):

    test_input_list = []
    extracted_signal_list = []
    for signal_index in range(num-6, num):
        print(signal_index)
        signal = data[signal_index]
        test_input = np.array(signal)
        test_input = torch.from_numpy(test_input)

        test_input = torch.unsqueeze(test_input, 0)
        print(test_input.shape)

        test_input = test_input.float().to(device)
        extracted_signal = model(test_input)

        test_input_value = test_input.cpu()
        test_input_value = test_input_value.detach().numpy()
        test_input_list = np.append(test_input_list, test_input_value[0])

        # test_output_value = test_output.cpu()
        # test_output_value = test_output_value.detach().numpy()
        # test_output_value = test_output_value[0]

        extracted_signal_value = extracted_signal.cpu()
        extracted_signal_value = extracted_signal_value.detach().numpy()
        extracted_signal_list = np.append(extracted_signal_list, extracted_signal_value[0])

    return extracted_signal_list



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


@st.cache
def load_raw_data(session_name='E01'):

    if session_name == 'E01':
        return 6
    elif session_name == 'E02':
        return 12
    elif session_name == 'E03':
        return 18
    elif session_name == 'E04':
        return 24
    elif session_name == 'E05':
        return 30
    elif session_name == 'E06':
        return 36
    elif session_name == 'E07':
        return 42
    else:
        return 48


def get_table_download_link(df):
    df = pandas.DataFrame(df)
    csv = df.to_csv(index=False)
    return csv


def main():
    test_input_np = np.load('test.npy')
    st.title("ЭЭГ аналитика")
    st.sidebar.subheader("")
    session_list = ['E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', 'E08']
    files_types = ['csv', 'tsv', ' xlsx', 'npy']
    session_name = st.sidebar.selectbox("Выбрать пример", session_list)

    num = load_raw_data(session_name)

    signals = []
    for signal_index in range(num-6, num):
        signals = np.append(signals, test_input_np[signal_index])
    signals_predict = signals.copy()
    signals = pd.DataFrame(signals)

    st.sidebar.subheader("Информация о ээг")
    freq_bands = ['альфа', 'бета', 'дельта', 'тета', 'гамма', 'все']
    selected_frequency = st.sidebar.selectbox("Выбор ритма", freq_bands)
    if selected_frequency == 'альфа':
        lowcut = 8
        highcut = 12
    elif selected_frequency == 'бета':
        lowcut = 13
        highcut = 30
    elif selected_frequency == 'дельта':
        lowcut = 0.5
        highcut = 4
    elif selected_frequency == 'тета':
        lowcut = 4
        highcut = 8
    elif selected_frequency == 'гамма':
        lowcut = 30
        highcut = 100
    elif selected_frequency == 'все':
        lowcut = 1
        highcut = 100

    sampled_channel = butter_bandpass_filter(signals.to_numpy(), lowcut, highcut, 256, order=6)

    if session_name and not file:
        st.write("Таблица сигналов ", session_name)
        st.dataframe(np.transpose(signals))
    selected_type = st.selectbox("Тип файла", files_types)
    st.write("Информация о ээг в ритме -  ", selected_frequency)
    st.line_chart(sampled_channel)
    st.write("Сигнал с обработкой")

    p = figure(
        title='EEG',
        x_axis_label='x',
        y_axis_label='y')

    p.line( [i for i in range (len(signals_predict))], signals_predict, legend_label='Input', line_color="red", line_width=2)
    p.line( [i for i in range (len(predict(test_input_np, num)))], predict(test_input_np, num), legend_label='Predict', line_width=2)
    st.bokeh_chart(p, use_container_width=True)


if __name__ == '__main__':
    main()
