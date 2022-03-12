import cv2

from kivy.app import App
from kivy.clock import Clock

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture

from deeplearning import face_mask_prediction


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.state = 'stop'
        self.capture = capture

        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        if self.state == 'stop':
            return
        ret, frames = self.capture.read()
        if ret == False:
            return

        image = face_mask_prediction(frames)

        # convert it to texture
        buf1 = cv2.flip(image, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(
            size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture


class FaceMaskApp(App):
    title = 'Face mask recognition software'

    def build(self):
        # create default layout
        layout = BoxLayout(orientation='vertical')

        self.label = Label(text='Face mask recognition software',
                           size_hint=(1, 0.2), font_size=24)

        # play/pause button
        self.play_button = Button(text='Play', size_hint=(
            1, 0.1), on_press=self.play_camera)

        # create a opencv camera object
        self.capture = cv2.VideoCapture(0)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

        # create a camera widget
        self.my_camera = KivyCamera(
            capture=self.capture, fps=24, size_hint=(1, 1))

        # add camera widget and play/stop button to the layout
        layout.add_widget(self.label)
        layout.add_widget(self.play_button)
        layout.add_widget(self.my_camera)
        return layout

    def play_camera(self, *args):
        if self.my_camera.state == 'play':
            self.my_camera.state = 'stop'
            self.play_button.text = 'Play'
        else:
            self.my_camera.state = 'play'
            self.play_button.text = 'Stop'

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    FaceMaskApp().run()