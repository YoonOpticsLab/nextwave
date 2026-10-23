import numpy as np
import midi

class nextwave_midi():
    def __init__(self,ui):
        self.ui = ui
        self.init_commands()
        self.midi1=midi.midi_control()
        if not (self.midi1 is None):
            self.midi1.init_midi(0)
        
    def close(self):
        self.midi1.close()
        
    def init_commands(self):
        fil=open("nextwave_midi.csv","rt")
        self.commands = {}
        for lin in fil.readlines():
            if lin[0]=="#":
                continue
            fields=lin.split(',')
            str_code = fields[3].strip().replace('~', ',') # COMMAS delimi CSV, but likely to need in code. Use Tilde instead, replace
            self.commands[int(fields[0])] = [fields[0], int(fields[1]), int(fields[2]), str_code  ]
        fil.close()
       
    def do_static(self):
        zs = np.zeros(65)
        zs[2:5] = [self.ui.static2, self.ui.static, self.ui.static4]
        zs[11] = self.ui.static12
        self.ui.engine.apply_static_mirror(zs)
                    
    def check_and_process(self):   
        if not(self.midi1 is None):
            msg=self.midi1.poll_knobs()
            if not (msg is None):
                which = msg[0]
                try:
                    command1 = self.commands[which]
                except:
                    print( "Unrecognized MIDI command")
                    return

                value = msg[1]
                if command1[1] == 1: # Knob
                    if value > 64:
                        value = 64 - value
                        value = - (2**abs(value)) + 1
                    else:
                        value =   (2**abs(value)) - 1
                        
                    #print( "Calling: ", command1[3], " with value=",value )
                    eval(command1[3])
                    #self.ui.signal_static_set.emit(3,value/10)
                    
                elif command1[1] == 2: # Button
                    if value==127: # Press
                        eval(command1[3])
