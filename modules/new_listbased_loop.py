class LoopI():

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.current_index = self.start
        self.the_end = False

    def step_up_index(self):
        if self.current_index < self.stop:
            self.current_index += 1
            self.the_end = False
            if self.current_index == self.stop:
                self.the_end = True

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, value):
        if value < self.stop:
            self.current_index = value
            self.the_end = False
        elif value == self.stop:
            self.current_index = value
            self.the_end = True
        else:
            self.the_end = True


class LoopIList():

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.sequence = [ i for i in range(stop)]
        self.sequence.insert(0, self.sequence[0])
        self.current_index = self.start
        self.the_end = False

    def step_up_index(self):
        if self.current_index < self.stop:
            self.current_index += 1
            self.the_end = False
            if self.current_index == self.stop:
                self.the_end = True

    def get_current_indexS(self):
        return self.sequence[self.current_index]

    def get_current_index(self):
        return self.current_index

    def set_current_index(self, value):
        if value < self.stop:
            self.current_index = value
            self.the_end = False
        elif value == self.stop:
            self.current_index = value
            self.the_end = True
        else:
            self.the_end = True


n = LoopIList(0, 50) # OUTER LOOP
k = LoopI(0, 560) # INNTER LOOP
PR = LoopIList(0, 5000000)
counter = 1

# n.step_up_index()
# n.step_up_index()

# print(n.get_current_index())

while True:


    if k.the_end:
        break

    # UK1[0] = complex(UM1[k,n,0], UM2[k,n,0])
    # UK1[1] = complex(UM1[k,n,1], UM2[k,n,1])
    # UK1[2] = complex(UM1[k,n,2], UM2[k,n,2])
    # UK1[3] = complex(0, 0)
    # AIK1[0] = complex(AIM1[k,n,0], AIM2[k,n,0])
    # AIK1[1] = complex(AIM1[k,n,1], AIM2[k,n,1])
    # AIK1[2] = complex(AIM1[k,n,2], AIM2[k,n,2])
    # AIK1[3] = complex(0, 0)

    if not (n.get_current_index() > 0) or (k.get_current_index() == 0 and PR.get_current_index() ==2):
        # print(counter); counter+=1
        pass
        # UK10=(UK1[0]+UK1[1]+UK1[2])/3
        # UK11=(UK1[0]+UK1[1]*AL+UK1[2]*AL**2)/3
        # UK12=(UK1[0]+UK1[1]*AL**2+UK1[2]*AL)/3
        # SKU2=np.sqrt(UK12.real**2+UK12.imag**2)/np.sqrt(UK11.real**2+UK11.imag**2)*100
        # SKU0=np.sqrt(UK10.real**2+UK10.imag**2)/np.sqrt(UK11.real**2+UK11.imag**2)*100
        # UK1[0]=UK11
        # UK1[1]=UK11*AL**2
        # UK1[2]=UK11*AL
        # AIK10=(AIK1[0]+AIK1[1]+AIK1[2])/3
        # AIK11=(AIK1[0]+AIK1[1]*AL+AIK1[2]*AL**2)/3
        # AIK12=(AIK1[0]+AIK1[1]*AL**2+AIK1[2]*AL)/3
        # SKI2=np.sqrt(AIK12.real**2+AIK12.imag**2)/np.sqrt(AIK11.real**2+AIK11.imag**2)*100
        # SKI0=np.sqrt(AIK10.real**2+AIK10.imag**2)/np.sqrt(AIK11.real**2+AIK11.imag**2)*100
        # AIK1[1]=AIK11
        # AIK1[2]=AIK11*AL**2
        # AIK1[3]=AIK11*AL



    # print("OUNTER, INNER", k.get_current_index(), n.get_current_indexS())
    # print(counter); counter+=1
    # RASCHET(k, n)


    if k.get_current_index() == 0 and PR.get_current_index() == 1:
        pass
    if k.get_current_index() == 0 and PR.get_current_index() == 2:
        pass
    if k.get_current_index() == 0 and PR.get_current_index() == 1:
        PR.step_up_index()
        continue
    if PR.get_current_index() == 2:
        PR.set_current_index(0)

    if n.the_end:
        k.step_up_index()
        n.set_current_index(0)
    else:
        n.step_up_index()

