import signal
import time

interrupted = False

def signal_handler(signal, frame):
  global interrupted
  interrupted = True

def prevent_interrupts():
  signal.signal(signal.SIGINT, signal_handler)

def was_interrupted():
  return interrupted

def interrupt_handled():
  global interrupted
  interrupted = False

if __name__ == '__main__':
  prevent_interrupts()
  for i in range(40):
    print('entering')
    time.sleep(5)
    print('exiting')
    if was_interrupted():
      print('Handling interrupt')
      interrupt_handled()