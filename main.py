import requests
import audio_memory_io
import numpy as np
from streamp3 import MP3Decoder
from io import BytesIO


class StreamDecoder:
    def __init__(self, url, timeout=20, chunk_size=200000) -> None:
        self.data = BytesIO()
        self.decoder = None
        self.streaming_data = requests.get(url, stream=True, timeout=timeout)
        self.chunk_size = chunk_size

    def update(self, data):
        p = self.data.tell()
        self.data.write(data)
        self.data.seek(p)

    def decode(self):
        if self.decoder is None:
            self.decoder = MP3Decoder(self.data)

        decoder_data = []
        for x in self.decoder:
            y = np.frombuffer(x, dtype="int16")/32768
            y = y.reshape(-1, self.decoder.num_channels)
            decoder_data.append(y)
        decoder_data = np.concatenate(decoder_data, axis=0)
        return decoder_data

    @property
    def sample_rate(self):
        return self.decoder.sample_rate

    def __iter__(self):
        for data in self.streaming_data.iter_content(chunk_size=self.chunk_size):
            self.update(data)
            yield self.decode()


def main(url):

    decoder_data = []
    for x in StreamDecoder(url):
        decoder_data.append(x)
        print(x.shape)

    decoder_data = np.concatenate(decoder_data, axis=0)
    audio_memory_io.save(np.array(decoder_data).T,
                         48000, "out.wav")


if __name__ == "__main__":
    main("https://cdn1.suno.ai/0ed0a3b5-6b00-4ac1-91c1-bd9494fa1138.mp3")
