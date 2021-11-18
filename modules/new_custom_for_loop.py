class LoopI():

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop
        self.current_index = self.start
        self.the_end = False

    def step_up_index(self):
        if self.current_index < self.stop:
            self.current_index += 1
            # self.the_end = False
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


# def mess_the_loop(reloop=False):
#     nonlocal PR, k, n
#     if reloop:
#         PR.step_up_index()
#     else:
#         if PR.get_current_index() == 2:
#             PR.set_current_index(1) # or 1 XZ

#         if k.the_end:
#             n.step_up_index()
#             k.set_current_index(0)
#         else:
#             k.step_up_index()




k = LoopI(0, 49) # INNER LOOP
n = LoopI(0, 560) # OUTER LOOP
PR = LoopI(1, 5)
counter = 1
# PR.step_up_index()


while True:


    if n.the_end:
        break

    K = k.get_current_index(); N = n.get_current_index();

    # UK1[0] = complex(UM1[N,K,0], UM2[N,K,0])
    # UK1[1] = complex(UM1[N,K,1], UM2[N,K,1])
    # UK1[2] = complex(UM1[N,K,2], UM2[N,K,2])
    # UK1[3] = complex(0, 0)
    # AIK1[0] = complex(AIM1[N,K,0], AIM2[N,K,0])
    # AIK1[1] = complex(AIM1[N,K,1], AIM2[N,K,1])
    # AIK1[2] = complex(AIM1[N,K,2], AIM2[N,K,2])
    # AIK1[3] = complex(0, 0)


    if not (k.get_current_index() > 0) or (k.get_current_index() == 0 and PR.get_current_index() == 2):
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



    # print(n.get_current_index(), k.get_current_index())
    # print(counter); counter+=1
    # RASCHET(N, K)


    if k.get_current_index() == 0 and PR.get_current_index() == 1:
        x=0
        # PPR1[N]=PP1
    if k.get_current_index() == 0 and PR.get_current_index() == 2:
        x=0
        # PPR2[N]=PP2
    if k.get_current_index() == 0 and PR.get_current_index() == 1:
        PR.step_up_index()
        continue

    if PR.get_current_index() == 2:
        PR.set_current_index(1)

    if k.the_end:
        n.step_up_index()
        k.set_current_index(0)
    else:
        k.step_up_index()

