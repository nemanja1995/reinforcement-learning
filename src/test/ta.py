import threading
from datetime import time
import time

from pynput.keyboard import Listener, Key


class KeyboardThread(threading.Thread):
    def __init__(self, name='keyboard-input-thread'):
        # self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        with Listener(
                on_press=self.on_press) as listener:
            listener.join()
            # waits to get input + Return

    def on_press(self, key):
        print('{0} pressed'.format(key))
        if key == Key.right:
            self.action = 0
        elif key == Key.left:
            self.action = 1

showcounter = 0
# something to demonstrate the change

# start the Keyboard thread
kthread = KeyboardThread()

while True:
    # the normal program executes without blocking. here just counting up
    showcounter += 1
    time.sleep(2)
    print("2 sec ...")
