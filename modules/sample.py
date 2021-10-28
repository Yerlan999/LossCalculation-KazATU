# 117 - 1700 - 1500
MM, MPR, MTR, DT, MT = 5, 3, 1, 2.5, 5
M = MPR + MTR
MMT = MM / MT

if M <= 6:
    M1 = M
if M > 6:
    M1 = 6

M10 = 2 * M
M20 = 4 * M

for n in range[560+1]:
    for k in range[49]:
        PR = 0
1700    CONTINUE
        PR = PR + 1

        UK1[0] = complex(UM1[n,k,0], UM2[n,k,0])
        UK1[1] = complex(UM1[n,k,1], UM2[n,k,1])
        UK1[2] = complex(UM1[n,k,2], UM2[n,k,2])
        UK1[3] = complex(0, 0)
        AIK1[0] = complex(AIM1[n,k,0], AIM2[n,k,0])
        AIK1[1] = complex(AIM1[n,k,1], AIM2[n,k,1])
        AIK1[2] = complex(AIM1[n,k,2], AIM2[n,k,2])
        AIK1[3] = complex(0, 0)

        if k > 0:
            GOTO 1111
        if k == 0 and PR == 2:
            GOTO 1111

        UK10=(UK1[0]+UK1[1]+UK1[2])/3
        UK11=(UK1[0]+UK1[1]*AL+UK1[2]*AL**2)/3
        UK12=(UK1[0]+UK1[1]*AL**2+UK1[2]*AL)/3
        SKU2=np.sqrt(UK12.real**2+UK12.imag**2)/np.sqrt(UK11.real**2+UK11.imag**2)*100
        SKU0=np.sqrt(UK10.real**2+UK10.imag**2)/np.sqrt(UK11.real**2+UK11.imag**2)*100
        UK1[0]=UK11
        UK1[1]=UK11*AL**2
        UK1[2]=UK11*AL
        AIK10=(AIK1[0]+AIK1[1]+AIK1[2])/3
        AIK11=(AIK1[0]+AIK1[1]*AL+AIK1[2]*AL**2)/3
        AIK12=(AIK1[0]+AIK1[1]*AL**2+AIK1[2]*AL)/3
        SKI2=np.sqrt(AIK12.real**2+AIK12.imag**2)/np.sqrt(AIK11.real**2+AIK11.imag**2)*100
        SKI0=np.sqrt(AIK10.real**2+AIK10.imag**2)/np.sqrt(AIK11.real**2+AIK11.imag**2)*100
        AIK1[1]=AIK11
        AIK1[2]=AIK11*AL**2
        AIK1[3]=AIK11*AL

1111    CONTINUE
        raschet(UK1,AIK1,k,n,PPP,PP1,PP2,ppp1,ppp2,ppp3,ppp4,ppp5,ppp6,ppp7,ppp8)

        if k == 0 and PR == 1:
            PPR1[n] = PP1
        if k == 0 and PR == 2:
            PPR2[n] = PP2
        if k == 0 and PR==1:
            GOTO 1700
        if PR == 2:
            GOTO 1500

1500    CONTINUE
