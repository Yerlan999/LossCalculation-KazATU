from time import sleep


n = 0
i = 0
il = ["a", "b", "c", "d"]


next_outer = False

update_inner = True

inner_loop = 10 # i
outer_loop = 4 # n
PR = 0


while True:

    PR += 1

    if i == outer_loop:
        break

    print("Outer,  Inner, PR: ", il[i], n, PR, update_inner)

    [block-1]

    if (i > 0) or (i == 0 and PR ==2):

        [block-2]

    [block-3]

    if update_inner:

        if n < inner_loop-1:
            n += 1
            next_outer = False
        else:
            n = 0
            next_outer = True

    if next_outer:
        if i < outer_loop:
            i += 1
        else:
            i = 0

    if i == 0 and PR == 1:
        update_inner = False  # ---> 1700
        continue
    if PR == 2:
        update_inner = True  # ---> 1500
        PR = 0
        continue


    sleep(1)
